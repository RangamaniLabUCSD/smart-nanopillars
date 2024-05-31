from dolfin import *
from smart import mesh_tools
import numpy as np

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

# Create mesh and define function space
nucRad = 3.0#5.5
thickness = 0.05
mesh_ref, mf2, mf3 = mesh_tools.create_spheres(outerRad=nucRad, innerRad=nucRad-thickness, 
                                               hEdge=0.2, hInnerEdge=0.2)
mesh = create_meshview(mf3, 1)
# mesh = BoxMesh(Point(-nucRad/2, -nucRad/2, 0.0), Point(nucRad/2, nucRad/2, thickness), 60, 60, 2)
npSpacing = 2.0
npRad = 0.5
xMax = 2.0 #np.floor((nucRad/2 - 2*npRad) / npSpacing) * npSpacing
xNP = np.arange(-xMax, xMax+1e-12, npSpacing)
yNP = np.arange(-xMax, xMax+1e-12, npSpacing)
xNP, yNP = np.meshgrid(xNP, yNP)
xNP = xNP.flatten()
yNP = yNP.flatten()


def compute_nanopillar_force(xNP, yNP, npRad, zTop, xTest, kRepel):
    Ftot = np.array([0, 0, 0])
    for i in range(len(xNP)):
        npDist = np.sqrt((xTest[0]-xNP[i])**2 + (xTest[1]-yNP[i])**2)
        if True:#npDist < (npRad + 2):
            if npDist < npRad:
                xyDist = 0
                dx = 0
                dy = 0
            else:
                xyDist = npDist - npRad
                dx = (xTest[0]-xNP[i])/npDist
                dy = (xTest[1]-yNP[i])/npDist
            if xTest[2] < zTop:
                distVal = xyDist
            else:
                distVal = np.sqrt(xyDist**2 + (xTest[2]-zTop)**2)
            if distVal == 0:
                raise ValueError("membrane intersects nanopillar")
            dx = dx*xyDist/distVal
            dy = dy*xyDist/distVal
            dz = (xTest[2]-zTop)/distVal
            Ftot = Ftot - (kRepel / distVal) * np.array([dx, dy, dz])
    return Ftot


# mf2 = MeshFunction("size_t", mesh_ref, 2)
# mf3 = MeshFunction("size_t", mesh_ref, 3, 1)

# mesh = UnitCubeMesh(24, 16, 16)
# mesh = create_meshview(mf3, 1)
# outer_surf = create_meshview(mf2, 10)
# inner_surf = create_meshview(mf2, 12)
File("nuc_indent/test_shell.pvd") << mesh
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# Mark boundary subdomains
top =  CompiledSubDomain("(x[2] > side) && on_boundary", side = nucRad*0.9)
# right = CompiledSubDomain("near(x[0], side) && on_boundary", side = nucRad/2)
# left = CompiledSubDomain("near(x[0], side) && on_boundary", side = -nucRad/2)
# front = CompiledSubDomain("near(x[1], side) && on_boundary", side = -nucRad/2)
# back = CompiledSubDomain("near(x[1], side) && on_boundary", side = nucRad/2)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0", "0.0"), degree=1)
# r = Expression(("-0.5", "0.0", "0.0"), degree=1)

bct = DirichletBC(V, c, top)
# bcr = DirichletBC(V, c, right)
# bcl = DirichletBC(V, c, left)
# bcf = DirichletBC(V, c, front)
# bcb = DirichletBC(V, c, back)

# bcs = [bcr, bcl, bcf, bcb]
bcs = [bct]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, 0.0, 0.0))  # Body force per unit volume
# T  = Constant((-0.001,  0.0, 0.0))  # Traction force on the boundary

domain_id = MeshFunction("size_t", mesh, 2, 0)
for f in facets(mesh):
    for i in range(len(xNP)):
        xCur = xNP[i]
        yCur = yNP[i]
        rCur = np.sqrt((f.midpoint().x()-xCur)**2 + (f.midpoint().y()-yCur)**2)
        RCur = np.sqrt(f.midpoint().x()**2 + f.midpoint().y()**2 + f.midpoint().z()**2)
        if RCur > nucRad-.01 and rCur <= npRad and f.midpoint().z() < -nucRad + 1.0: #np.isclose(f.midpoint().z(),0.0)  and rCur <= npRad:
            domain_id[f] = 1
ds = Measure('ds', domain=mesh, subdomain_data=domain_id)

# Kinematics
d = u.geometric_dimension()
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
I1 = tr(C)
I2 = 0.5*(I1**2 - tr(C*C))
J  = det(F)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
E1, E2 = Constant(5000.0), Constant(1000.0)

# Stored strain energy density (incompressible Mooney Rivlin model)
# psi = (mu/2)*(I1 - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
psi = E1*(I1-3) + E2*(I2-3) + 1e6*(J-1)**2

zNP = [-0.05]#[-nucRad - 0.05]
idx = 0
zMove = 0.1
zStep = 0.01
zFinal = zNP[-1] + zMove
u_file = XDMFFile("nuc_indent/test_np_sphere.xdmf")
u_file.parameters["flush_output"] = True
u_file.write(u, idx)
x = SpatialCoordinate(mesh)
xCur = x
kMin = 0.1
kRepel = 0.2#10.0
kRamp = [kMin]

zIndent = 0.5

while u(0,0,-nucRad)[2] < zIndent:#zNP[-1] < zFinal-1e-6:

    # exprStr = ""
    # x_list = [-0.5, 0, 0.5, -0.5, 0, 0.5, -0.5, 0, 0.5]
    # y_list = [0.5, 0.5, 0.5, 0, 0, 0, -0.5, -0.5, -0.5]
    # z_list = [-1.87, -1.94, -1.87, -1.94, -2, -1.94, -1.87, -1.94, -1.87]
    # for i in range(len(x_list)):
    #     exprStr += f"100.0*exp(-(pow(x[0]-({x_list[i]}),2) + pow(x[1]-({y_list[i]}),2) + pow(x[2]-({z_list[i]}),2))/0.05)"
    #     if i < len(x_list)-1:
    #         exprStr += " + "

    # xNew = x + u
    # xNP = 0
    # yNP = 0
    # npDist = ufl.sqrt((xNew[0]-xNP)**2 + (xNew[1]-yNP)**2)
    # xyDist = npDist - npRad
    # dx_np = (xNew[0]-xNP)/xyDist
    # dy_np = (xNew[1]-yNP)/xyDist
    # zDistTest = xNew[2]-(-nucRad-zNP[-1])
    # zDist = zDistTest*(1+ufl.sign(zDistTest))/2
    # distVal = ufl.sqrt(xyDist**2 + zDist**2)
    # dz_np = zDist/distVal
    # forceMag = kRepel/distVal
    # xForce = -forceMag * dx_np
    # yForce = -forceMag * dy_np
    # zForce = -forceMag * dz_np
    
    # curForce = kRepel/(-nucRad + u(0,0,-nucRad)[2]-zNP[-1])
    # curForce = kRepel#/(u(0,0,0)[2]-zNP[-1])
    # print(f"Current force is {curForce}")
    # T = Expression(("0.0", "0.0", "curForce"), degree=1, curForce=curForce)
    try:
        T = Function(V)
        coords = V.tabulate_dof_coordinates()
        Tvec = T.vector().get_local()
        for i in range(0,len(coords),3):
            Tvec[i:i+3] = compute_nanopillar_force(xNP, yNP, npRad, -nucRad+zNP[-1], coords[i,:], kRamp[-1])
        T.vector().set_local(Tvec)
        T.vector().apply("insert")
    except:
        print("Nanopillar ran into nanopillar, reset position")
        zNP[-1] = (zNP[-1]+zNP[-2])/2
        continue

    # Total potential energy
    
    Pi = psi*dx - dot(B, u)*dx + dot(T,u)*ds #(u[2]*kRepel/(u[2]-zNP[-1]))*ds(1) #dot(T, u)*ds(1)

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = derivative(Pi, u, v)

    # Compute Jacobian of F
    J = derivative(F, u, du)

    # # Solve variational problem
    # try:
    #     solve(F == 0, u, bcs, J=J,
    #         form_compiler_parameters=ffc_options)
    # except:
    #     zNP[-1] = zNP[-2] + 0.5*(zNP[-1]-zNP[-2])
    #     continue
    problem = NonlinearVariationalProblem(F, u, bcs, J=J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1E-8
    prm["newton_solver"]["relative_tolerance"] = 1E-6
    prm["newton_solver"]["maximum_iterations"] = 100
    try:
        solver.solve()
        if kRamp[-1] < kRepel:
            kRamp.append(min([kRamp[-1]+0.1, kRepel]))
            continue
        elif kRamp[-1] == kRepel:
            kRamp = [kMin]
        else:
            raise ValueError("k cannot be greater than kRepel")
    except:
        print(f"Resetting kRamp from {kRamp[-1]} to")
        if len(kRamp) == 1:
            kRamp[-1] = kRamp[-1]/2
        else:
            kRamp[-1] = (kRamp[-1]+kRamp[-2])/2
        print(f"{kRamp[-1]} because solve failed")
        continue

    idx += 1
    print(f"Done computing idx = {idx}")
    u_file.write(u, idx)
    # zNP.append((zNP[-1]+u(0,0,0)[2])/2)
    # kRepel += 0.01
    zNP.append(zNP[-1]+zStep)
    xCur = xCur + u
    np.savetxt("zNP.txt", np.array(zNP))
