"""
Functions to create meshes for cell mechanotransduction on nanopillars
"""

from typing import Tuple
import pathlib
import numpy as np
import dolfin as d
from mpi4py import MPI
from sympy.parsing.sympy_parser import parse_expr
from smart.mesh_tools import implicit_curve, compute_curvature, gmsh_to_dolfin, facet_topology

def get_shape_coords(contactRad, nanopillars, nuc_compression):
    aVecRef = np.array([35, 2000, 1000, 1200, 1500, 1000])
    bVecRef = np.array([0.94, 0.4, 0.4, 0.4, 0.4, 0.4])
    cVecRef = np.array([0.01, 9.72, 15, 24.5, 33.2, 42])
    dVecRef = np.array([30, 15, 10, 1.0, 0.12, 0.0322])
    RList = np.array([10, 13, 15, 20, 25, 30])
    scaleFactor = 0.82
    nucScaleFactor = 0.8
    aVecRef = aVecRef / scaleFactor**4
    cVecRef = cVecRef / scaleFactor
    dVecRef = dVecRef / scaleFactor**4
    RList = RList / scaleFactor
    findMatch = np.isclose(RList, contactRad)
    if np.any(findMatch):
        idx = np.nonzero(findMatch)[0][0]
        refParam = [aVecRef[idx], bVecRef[idx], cVecRef[idx], dVecRef[idx], RList[idx]]
    elif contactRad < max(RList) and contactRad > min(RList):
        aVal = np.interp(contactRad, np.array(RList), np.array(aVecRef))
        bVal = np.interp(contactRad, np.array(RList), np.array(bVecRef))
        cVal = np.interp(contactRad, np.array(RList), np.array(cVecRef))
        dVal = np.interp(contactRad, np.array(RList), np.array(dVecRef))
        refParam = [aVal, bVal, cVal, dVal, contactRad]
    else:
        raise ValueError("This shape is outside the specified range")

    nuc_vol = (4/3)*np.pi*5.3*5.3*2.4/nucScaleFactor**3
    nanopillar_rad = nanopillars[0]
    nanopillar_height = nanopillars[1]
    nanopillar_spacing = nanopillars[2]
    if np.all(np.array(nanopillars) != 0):
        xSteric = 0.05
        xCurv = 0.2
        num_pillars = 2*np.ceil(contactRad/nanopillar_spacing) + 1
        rMax = nanopillar_spacing * np.ceil(contactRad/nanopillar_spacing)
        test_coords = np.linspace(-rMax, rMax, int(num_pillars))
        xTest, yTest = np.meshgrid(test_coords, test_coords)
        rTest = np.sqrt(xTest**2 + yTest**2)
        keep_logic = rTest <= contactRad-nanopillar_rad-xCurv-xSteric
        xTest, yTest = xTest[keep_logic], yTest[keep_logic]
        num_pillars_tot = len(xTest)
        nanopillar_vol = num_pillars_tot * np.pi*(
            (nanopillar_rad+xSteric)**2 * (nanopillar_height-xSteric-xCurv) +
            xSteric*(nanopillar_rad**2 + nanopillar_rad*xSteric*np.pi/2 + 2*xSteric**2/3) +
            xCurv*((nanopillar_rad+xSteric+xCurv)**2 - (nanopillar_rad+xSteric+xCurv)*xCurv*np.pi/2 + 
                   2*xCurv**2/3))
        zOffset = nanopillar_height
    else:
        nanopillar_vol = 0
        zOffset = 0
    
    targetVol = 480/scaleFactor**3 *4 + nuc_vol + nanopillar_vol
    # rValsOuter, zValsOuter = implicit_curve(make_expr(*refParam))
    rValsOuter, zValsOuter, outParam = shape_adj_axisymm(refParam, zOffset, targetVol)
    rValsOuter, zValsOuter = dilate_axisymm(rValsOuter, zValsOuter, targetVol, 0.0)

    zMax = max(zValsOuter)
    innerExpr = get_inner(zOffset, zMax, nucScaleFactor, nuc_compression)
    aInner, bInner, r0Inner, z0Inner = get_inner_param(zOffset, zMax, nucScaleFactor, nuc_compression)
    innerParam = [aInner, bInner, r0Inner, z0Inner]
    if not innerExpr == "" and np.all(np.array(nanopillars) != 0):
        u_nuc, aInner, bInner = get_u_nuc(zOffset, zMax, nucScaleFactor, nuc_compression, nanopillars)
        aMod = aInner - innerParam[0]
        bMod = bInner - innerParam[1]
        # update inner coordinates according to uEllipsoid
        innerParam[0] += aMod
        innerParam[1] += bMod
        assert bMod == 0
        innerExpr = get_inner(zOffset, zMax, nucScaleFactor, nuc_compression, aMod, bMod)
        if nuc_compression > 0:
            rValsInner, zValsInner = implicit_curve(innerExpr, num_points=501)
        else:
            rValsInner, zValsInner = implicit_curve(innerExpr, num_points=101)
        rScale = (aInner + aMod) / aInner
    elif not innerExpr == "":
        rValsInner, zValsInner = implicit_curve(innerExpr, num_points=101)
        u_nuc = []
        rScale = 1
    else:
        rValsInner = []
        zValsInner = []
        innerParam = []
        u_nuc = []
        rScale = 1

    return [rValsOuter, zValsOuter, rValsInner, zValsInner, innerParam, u_nuc, rScale, outParam]

def get_u_nuc(zOffset, zMax, nucScaleFactor, nuc_compression, nanopillars):
    nanopillar_rad, nanopillar_height, nanopillar_spacing = nanopillars[:]
    xSteric = 0.05
    nanopillar_rad += 2*xSteric # to avoid collision with PM
    aInner0, bInner, r0Inner, z0Inner = get_inner_param(zOffset, zMax, nucScaleFactor, nuc_compression)
    aInner = aInner0
    # define nanopillar locations
    xMax = np.ceil(aInner / nanopillar_spacing) * nanopillar_spacing
    xNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
    yNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
    xNP, yNP = np.meshgrid(xNP, yNP)
    xNP = xNP.flatten()
    yNP = yNP.flatten()

    indentation = nuc_compression
    zNP = -bInner + indentation

    def np_displ(coords, xNP, yNP, zNP, radNP):
        if len(zNP) == 1:
            zNP = zNP * np.ones_like(xNP)
        else:
            assert len(zNP) == len(xNP)
        dist_vals = np.sqrt(np.power(xNP-coords[0], 2) + np.power(yNP-coords[1], 2)) 
        uVal = 0
        for i in range(len(xNP)):
            uRef = zNP[i] - coords[2]
            if uRef > 0:
                if dist_vals[i] < radNP:
                    uVal = uRef
                    break
                else:
                    uVal += uRef*np.exp(-((dist_vals[i]-radNP)/0.2)**2)
        return np.array([0, 0, uVal])

    from smart import mesh_tools
    mesh_ellipsoid, mf2, mf3 = mesh_tools.create_ellipsoids(outerRad=[aInner,aInner,bInner], innerRad=[0,0,0], 
                                                hEdge=0.1, hInnerEdge=0.5)
    
    mesh_bound = d.create_meshview(mf2, 10)
    V_full = d.FunctionSpace(mesh_ellipsoid, d.VectorElement("P", mesh_ellipsoid.ufl_cell(), degree = 1, dim = 3))
    u_full = d.Function(V_full)
    coords_full = V_full.tabulate_dof_coordinates()
    uvec = u_full.vector()[:]
    vol_error = 100
    ref_vol = d.assemble(1.0 * d.Measure("dx", mesh_ellipsoid))
    vol_vec = []
    alpha_vec = [1]
    u_ellipsoid = d.Function(V_full)
    while vol_error > 0.01:
        if vol_error < 100: # not first iteration
            if len(alpha_vec) == 1:
                alpha_vec.append(np.sqrt(ref_vol/vol_vec[-1]))
            else:
                alpha_vec.append(alpha_vec[-2] + 
                                 (ref_vol-vol_vec[-2])*(alpha_vec[-1]-alpha_vec[-2])/(vol_vec[-1]-vol_vec[-2]))
            aInner = aInner0*alpha_vec[-1]
            u_ellipsoid = d.interpolate(d.Expression(("x[0]*(alpha-1)", "x[1]*(alpha-1)", "0.0"), 
                                            z0 = z0Inner, alpha=alpha_vec[-1], degree = 1), V_full)
            coords_full = V_full.tabulate_dof_coordinates()
            u_ellipsoid_vec = u_ellipsoid.vector()[:]
            for i in range(0, len(coords_full), 3):
                coords_full[i] += u_ellipsoid_vec[i:i+3]
                coords_full[i+1] += u_ellipsoid_vec[i:i+3]
                coords_full[i+2] += u_ellipsoid_vec[i:i+3]
        for i in range(0,len(coords_full),3):
            # if (np.sqrt(coords_full[i,0]**2 + coords_full[i,1]**2) < aInner and
            #     coords_full[i,2] < 0):
            #     uvec[i:i+3] = np_displ(coords_full[i,:], xNP, yNP, [zNP], nanopillar_rad)
            # else:
            #     uvec[i:i+3] = [0,0,0]
            uvec[i:i+3] = np_displ(coords_full[i,:], xNP, yNP, [zNP], nanopillar_rad)
            dist_vals = np.sqrt(np.power(xNP-coords_full[i,0], 2) + np.power(yNP-coords_full[i,1], 2))
            if min(dist_vals) < nanopillar_rad and min(dist_vals) > nanopillar_rad-xSteric and coords_full[i,2] < zNP:
                assert uvec[i+2] > 0.99*(zNP-coords_full[i,2])
            
        uvec = uvec + u_ellipsoid.vector()[:]
        u_full.vector().set_local(uvec)
        u_full.vector().apply("insert")
        mesh_copy = d.Mesh(mesh_ellipsoid)
        d.ALE.move(mesh_copy, u_full)
        vol_vec.append(d.assemble(1.0 * d.Measure("dx", mesh_copy)))
        vol_error = 100 * np.abs(vol_vec[-1] - ref_vol) / ref_vol

    V_bound =  d.FunctionSpace(mesh_bound, d.VectorElement("P", mesh_bound.ufl_cell(), degree = 1, dim = 3))
    u_bound = d.Function(V_bound)
    uvec = u_bound.vector()[:]
    bound_coord = V_bound.tabulate_dof_coordinates()
    for i in range(0, len(uvec), 3):
        cur_coord = bound_coord[i]
        uvec[i:i+3] = u_full(cur_coord)
    u_bound.vector().set_local(uvec)
    u_bound.vector().apply("insert")
    u_bound.set_allow_extrapolation(True)
    # ellipsoid_deform = np.reshape(u_ellipsoid.vector()[:],[-1,3])
    aMod = aInner - aInner0 #max(ellipsoid_deform[:,0])
    bMod = 0.0 #max(ellipsoid_deform[:,2])
    # u_shift = d.project(d.Expression(("0.0", "0.0", "z0"), z0 = z0Inner, degree = 1), V_bound)
    # d.ALE.move(mesh_bound, u_shift)
    return (u_bound, aInner, bInner)

def compute_stretch(u: d.Function, aInner: float, bInner: float, nanopillars: list, z0Inner: float):
    nanopillar_spacing = nanopillars[2]
    xMax = np.ceil(aInner / nanopillar_spacing) * nanopillar_spacing
    xNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
    yNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
    xNP, yNP = np.meshgrid(xNP, yNP)
    xNP = xNP.flatten()
    yNP = yNP.flatten()
    
    I = d.Identity(3)
    F = I + d.grad(u)    # Deformation gradient
    C = F.T*F          # Right Cauchy-Green deformation tensor
    J = d.det(F)
    mesh_bound = u.function_space().mesh()
    V_vector = d.FunctionSpace(mesh_bound, d.VectorElement("P", mesh_bound.ufl_cell(), degree = 1, dim = 3))
    V_scalar = d.FunctionSpace(mesh_bound, "P", 1)
    normal_expr = d.Expression(("(x[0]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2],2)/pow(b,4))", 
                                "(x[1]/pow(a,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2],2)/pow(b,4))", 
                                "(x[2]/pow(b,2))/sqrt((pow(x[0],2)+pow(x[1],2))/pow(a,4) + pow(x[2],2)/pow(b,4))"),
                                a=aInner, b=bInner, degree=1)
    normals = d.project(normal_expr, V_vector)
    a_vector = J * d.dot(normals, d.inv(F))
    a_vector = d.project(a_vector, V_vector)
    a_scalar = d.Function(V_scalar)
    V_scalar_coords = V_scalar.tabulate_dof_coordinates()
    avec = a_scalar.vector()[:]
    for i in range(len(avec)):
        cur_coord = V_scalar_coords[i]
        avec[i] = calc_stretch_NP(cur_coord[0],cur_coord[1],cur_coord[2]+z0Inner,
                 aInner,bInner,z0Inner,xNP,yNP,nanopillars)
        # a_cur = a_vector(V_scalar_coords[i])
        # avec[i] = np.sqrt(a_cur[0]**2 + a_cur[1]**2 + a_cur[2]**2)
    a_scalar.vector().set_local(avec)
    a_scalar.vector().apply("insert")
    a_nuc = a_scalar
    a_nuc.set_allow_extrapolation(True)
    return a_nuc

def create_3dcell(
    contactRad: float = 10.0,
    hEdge: float = 0,
    hInnerEdge: float = 0,
    hNP: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_vol_tag: int = 2,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
    return_curvature: bool = False,
    nanopillars: Tuple[float, float, float] = [0, 0, 0],
    thetaExpr: str = "",
    use_tmp: bool = False,
    roughness: Tuple[float, float] = [0, 0],
    sym_fraction: float = 1.0,
    nuc_compression: float = 0.0,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a 3d cell mesh.
    The inner contour (e.g. nucleus or other organelle) is defined
    implicitly through innerExpr, which is rotated about the z axis
    to form an axisymmetric shape (e.g. unit sphere centered at (0, 2) 
    defined by innerExpr = "r**2 + (z-2)**2 - 1")
    The outer cell contour is defined in terms of 
    cylindrical coordinates r, z, and theta.
    It is assumed that r can be expressed as a function of z and theta.
    If r = r1(z)T(theta), r1 is defined implicitly by outerExpr or innerExpr 
    (e.g. circle with radius 5 defined by outerExpr = "r**2 + z**2 - 25")
    and T is defined by thetaExpr (for an axisymmetric geometry, thetaExpr = "1")
    It is assumed that substrate is present at z = 0, so if the curve extends
    below z = 0 , there is a sharp cutoff.
    
    Some relevant examples that include theta dependence:
    * thetaExpr = "T0 + T1*cos(5*theta)", where T0 and T1 are user-defined numbers, 
        describes a five-pointed star geometry
    * thetaExpr = "a*b/sqrt((b*cos(theta))**2 + (a*sin(theta))**2)", where a and b
        are user-defined numbers, describes an ellipse contact region
    
    Special cases:
    * thetaExpr = "rectAR", where AR is a number representing an aspect ratio,
        defines a rectangular geometry at the cell contact region.
    * thetaExpr not given or thetaExpr = "" -> define axisymmetric 3d geometry by
        rotating the gmsh object

    Args:
        outerExpr: String implicitly defining an r-z curve for the outer surface
        innerExpr: String implicitly defining an r-z curve for the inner surface
        hEdge: maximum mesh size at the outer edge
        hInnerEdge: maximum mesh size at the edge
            of the inner compartment
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on the outer ellipsoid with
        inner_vol_tag: The value to mark the inner ellipsoidal volume with
        outer_vol_tag: The value to mark the outer ellipsoidal volume with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
        return_curvature: If true, return curvatures as a vertex mesh function
    Returns:
        Tuple (mesh, facet_marker, cell_marker)
    Or, if return_curvature = True, Returns:
        Tuple (mesh, facet_marker, cell_marker, curvature_marker)
    """
    import gmsh
    
    nanopillar_rad, nanopillar_height, nanopillar_spacing = nanopillars[:]
    zOffset = nanopillar_height
    xSteric = 0.05
    xCurv = 0.2 
    rValsOuter, zValsOuter, rValsInner, zValsInner, innerParams, u_nuc, rScale, outParam = get_shape_coords(
                                            contactRad, nanopillars, nuc_compression)

    rValsOuterClosed = np.concatenate((rValsOuter, -rValsOuter[::-1]))
    zValsOuterClosed = np.concatenate((zValsOuter,  zValsOuter[::-1]))
    curvFcnOuter = compute_curvature_1D(rValsOuterClosed, zValsOuterClosed, 
                                   curvRes=0.1, incl_parallel=True)
    if not len(rValsInner) == 0:
        aInner, bInner, r0Inner, z0Inner = innerParams[:]
        zMid = np.mean(zValsInner)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        RInnerVec = np.sqrt(rValsInner**2 + (zValsInner - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
        maxInnerDim = max(RInnerVec)
    else:
        zMid = np.mean(zValsOuter)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
    
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * maxOuterDim
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.2 * maxOuterDim if len(rValsInner) == 0 else 0.2 * maxInnerDim
    if np.isclose(hNP, 0):
        hNP = 0.1 * maxOuterDim
    # Create the two axisymmetric body mesh using gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("3dcell")

    # first add outer body and revolve
    thetaVec = [0]
    dtheta = hEdge/max(rValsOuter)
    cell_plane_tag = []
    bottom_point_list = []
    outer_spline_list = []
    edge_surf_list = []
    edge_segments = []
    top_point = gmsh.model.occ.add_point(0.0, 0.0, zValsOuter[0])
    # rotate shape 2*pi in the case of no theta dependence
    outer_tag_list = []
    line_tag_list = []
    for i in range(len(rValsOuter)):
        if i == 0:
            outer_tag_list.append(top_point)
        else:
            cur_tag = gmsh.model.occ.add_point(rValsOuter[i], 0.0, zValsOuter[i])
            line_tag = gmsh.model.occ.add_line(cur_tag, outer_tag_list[-1])
            line_tag_list.append(line_tag)
            outer_tag_list.append(cur_tag)
    outer_spline = gmsh.model.occ.add_spline(outer_tag_list)
    origin_tag = gmsh.model.occ.add_point(0, 0, 0)
    symm_axis_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[0])
    bottom_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[-1])
    outer_loop_tag = gmsh.model.occ.add_curve_loop(
        [outer_spline, symm_axis_tag, bottom_tag]
    )
    cell_plane_tag = gmsh.model.occ.add_plane_surface([outer_loop_tag])
    outer_shape = gmsh.model.occ.revolve([(2, cell_plane_tag)], 0, 0, 0, 0, 0, 1, 2*np.pi*sym_fraction)
    
    if np.all(np.array(nanopillars) != 0):
        zero_idx = np.nonzero(zValsOuter <= 0.0)
        num_pillars = 2*np.ceil(rValsOuter[zero_idx]/nanopillar_spacing) + 1
        rMax = nanopillar_spacing * np.ceil(rValsOuter[zero_idx]/nanopillar_spacing)
        test_coords = np.linspace(-rMax[0], rMax[0], int(num_pillars[0]))
        xTest, yTest = np.meshgrid(test_coords, test_coords)
        rTest = np.sqrt(xTest**2 + yTest**2)
        thetaTest = np.arctan2(yTest, xTest)
        if sym_fraction==1:
            keep_logic = rTest <= rValsOuter[zero_idx]-nanopillar_rad-xCurv-xSteric
        else:
            keep_logic1 = rTest <= rValsOuter[zero_idx]-nanopillar_rad-xCurv-xSteric
            keep_logic2 = np.logical_and(thetaTest < (2*np.pi*sym_fraction + np.pi/100), thetaTest > (0.0-np.pi/100))
            keep_logic = np.logical_and(keep_logic1, keep_logic2)
        xTest, yTest = xTest[keep_logic], yTest[keep_logic]
        for i in range(len(xTest)):
            # add points and then rotate?
            np_tag_list = []
            np_line_list = []
            zCur = nanopillar_height + xSteric
            dtheta = np.pi/8
            thetaCurv1 = np.arange(np.pi/2, 0, -dtheta)
            xCurv1 = nanopillar_rad + xSteric*np.cos(thetaCurv1)
            zCurv1 = nanopillar_height + xSteric*(np.sin(thetaCurv1)-1)
            thetaCurv2 = np.arange(np.pi, 3*np.pi/2, dtheta)
            xCurv2 = nanopillar_rad + xSteric + xCurv*(1+np.cos(thetaCurv2))
            zCurv2 = xCurv*(1+np.sin(thetaCurv2))
            xCurVec = xTest[i] + np.concatenate((np.array([0]), xCurv1, 
                                                np.array([nanopillar_rad+xSteric]), xCurv2, 
                                                np.array([nanopillar_rad+xSteric+xCurv])))
            yCurVec = yTest[i] * np.ones([len(xCurVec),1])
            zCurVec = np.concatenate((np.array([nanopillar_height]), zCurv1, 
                                     np.array([nanopillar_height-xSteric]), zCurv2, 
                                     np.array([0])))
            for j in range(len(zCurVec)):
                cur_tag = gmsh.model.occ.add_point(xCurVec[j], yCurVec[j], zCurVec[j])
                if j > 0:
                    cur_line = gmsh.model.occ.add_line(cur_tag, np_tag_list[-1])
                    np_line_list.append(cur_line)
                np_tag_list.append(cur_tag)

            # np_spline_tag = gmsh.model.occ.add_spline(np_tag_list)
            origin_np_tag = gmsh.model.occ.add_point(xTest[i], yTest[i], 0)
            bottom_np_tag = gmsh.model.occ.add_line(origin_np_tag, np_tag_list[-1])
            symm_np_tag = gmsh.model.occ.add_line(np_tag_list[0], origin_np_tag)
            np_loop_tag = gmsh.model.occ.add_curve_loop([*np_line_list, bottom_np_tag, symm_np_tag])
            np_plane_tag = gmsh.model.occ.add_plane_surface([np_loop_tag])
            np_shape = gmsh.model.occ.revolve([(2, np_plane_tag)], xTest[i], yTest[i], 0,
                                              0, 0, 1, 2 * np.pi)
            np_shape_tags = []
            for j in range(len(np_shape)):
                if np_shape[j][0] == 3:  # pull out tags associated with 3d objects
                    np_shape_tags.append(np_shape[j][1])
            assert len(np_shape_tags) == 1  # should be just one 3D body from the full revolution
            # cyl_tag = gmsh.model.occ.add_cylinder(
            #     xTest[i], yTest[i], 0.0, 0, 0, nanopillar_height-nanopillar_rad, nanopillar_rad)    
            # cap_tag = gmsh.model.occ.add_sphere(
            #     xTest[i], yTest[i], nanopillar_height-nanopillar_rad, nanopillar_rad)
            # cur_pillar = gmsh.model.occ.fuse([(3,cyl_tag)], [(3,cap_tag)])
            # print(np_shape[1])
            (outer_shape, outer_shape_map) = gmsh.model.occ.cut(outer_shape, [(3, np_shape_tags[0])])
            outer_shape_list = []
            for j in range(len(outer_shape_map)):
                if outer_shape_map[j]!=[]:
                    outer_shape_list.append(outer_shape_map[j][0])
            outer_shape = outer_shape_list

    outer_shape_tags = []
    for i in range(len(outer_shape)):
        if outer_shape[i][0] == 3:  # pull out tags associated with 3d objects
            outer_shape_tags.append(outer_shape[i][1])
    assert len(outer_shape_tags) == 1  # should be just one 3D body from the full revolution

    if len(rValsInner) == 0:
        # No inner shape in this case
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(3, outer_shape_tags, tag=outer_vol_tag)
        facets = gmsh.model.getBoundary([(3, outer_shape_tags[0])])
        # facets = gmsh.model.getBoundary(outer_shape)
        # assert (
        #     len(facets) == 2
        # )  # 2 boundaries because of bottom surface at z = 0, both belong to PM
        facet_tags = []
        for i in range(len(facets)):
            facet_tags.append(facets[i][1])
        gmsh.model.add_physical_group(2, facet_tags, tag=outer_marker)
    else:
        # Add inner shape
        # inner_tag_list = []
        # inner_line_list = []
        # for i in range(len(rValsInner)):
        #     if u_nuc == []:
        #         uCur = [0, 0, 0]
        #     else:
        #         uCur = u_nuc(rValsInner[i], 0., zValsInner[i]-z0Inner)
        #     # apply ellipse deformation
        #     cur_tag = gmsh.model.occ.add_point(rValsInner[i]+uCur[0], 0., zValsInner[i])
        #     inner_tag_list.append(cur_tag)
        #     if i > 0:
        #         cur_line = gmsh.model.occ.add_line(inner_tag_list[-2], inner_tag_list[-1])
        #         inner_line_list.append(cur_line)
        # inner_spline_tag = gmsh.model.occ.add_spline(inner_tag_list)
        # symm_inner_tag = gmsh.model.occ.add_line(inner_tag_list[0], inner_tag_list[-1])
        # inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, symm_inner_tag])
        # inner_plane_tag = gmsh.model.occ.add_plane_surface([inner_loop_tag])
        # inner_shape = gmsh.model.occ.revolve([(2, inner_plane_tag)], 0, 0, 0, 0, 0, 1, 2*np.pi*sym_fraction)

        if np.all(np.array(nanopillars) != 0):
            xMax = np.ceil(aInner / nanopillar_spacing) * nanopillar_spacing
            xNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
            yNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
            xNP, yNP = np.meshgrid(xNP, yNP)
            xNP = xNP.flatten()
            yNP = yNP.flatten()
            inner_spline_list = []
            # line_list_list = []
            edge_surf_list = []
            # cut_vol_list = []
            # first_points_list = []
            if u_nuc == []:
                u_top, u_bottom = 0, 0 #[0.,0.,0.], [0.,0.,0.]
            else:
                u_top = 0.0 #u_nuc(0.0, 0.0, zValsInner[0]-z0Inner)
                u_bottom = calc_def_NP(0.0, 0.0, zValsInner[-1], aInner, bInner, z0Inner, xNP, yNP, nanopillars)
            top_point = gmsh.model.occ.add_point(0.0, 0.0, zValsInner[0]+u_top)
            bottom_point = gmsh.model.occ.add_point(0.0, 0.0, zValsInner[-1]+u_bottom)
            num_theta = np.ceil(321*sym_fraction)
            thetaVecInner = np.linspace(0, 2*np.pi*sym_fraction, int(num_theta))
            sValsInner = np.zeros_like(rValsInner)
            for i in range(1,len(rValsInner)): # define arc length
                sValsInner[i] = sValsInner[i-1] + np.sqrt((rValsInner[i]-rValsInner[i-1])**2 + 
                                                          (zValsInner[i]-zValsInner[i-1])**2)
            profileList = []
            for j in range(len(thetaVecInner)):
                thetaCur = thetaVecInner[j]
                if j == (len(thetaVecInner)-1) and sym_fraction == 1:
                    inner_spline_list.append(inner_spline_list[0])
                    # first_points_list.append(first_points_list[0])
                    # line_list_list(line_list_list[0])
                else:
                    inner_tag_list = []
                    rValsInnerCur = rValsInner.copy()
                    zValsInnerCur = zValsInner.copy()
                    sValsInnerCur = sValsInner.copy()
                    # first, compute current deformations based on u_nuc
                    for i in range(len(rValsInnerCur)):
                        xRef = rValsInnerCur[i]*np.cos(thetaCur)
                        yRef = rValsInnerCur[i]*np.sin(thetaCur)
                        zRef = zValsInnerCur[i]
                        if u_nuc == []:
                            uCur = 0#[0, 0, 0]
                        else:
                            uCur = calc_def_NP(xRef, yRef, zRef, aInner, bInner, z0Inner, xNP, yNP, nanopillars)
                            # uCur = u_nuc(xRef/rScale, yRef/rScale, zRef-z0Inner)
                        zValsInnerCur[i] += uCur#[2]
                        if i > 0:
                            dsCur = np.sqrt((rValsInnerCur[i]-rValsInnerCur[i-1])**2 +
                                            (zValsInnerCur[i]-zValsInnerCur[i-1])**2)
                            sValsInnerCur[i] = sValsInnerCur[i-1] + dsCur
                    half_idx = int(np.floor(len(rValsInnerCur)/2))
                    sFirstHalf = np.linspace(0.0, sValsInnerCur[half_idx], 21)
                    sValsCur = sFirstHalf
                    dsBig = sValsInner[-1] / 200
                    dsSmall = sValsInner[-1] / 200
                    # determine where we are in theta direction - where do we potentially intersect nanopillars?
                    sortIdx = np.argsort(xTest)
                    xTest = xTest[sortIdx]
                    yTest = yTest[sortIdx]
                    rCross = []
                    for n in range(len(xTest)):
                        dCur = np.sqrt((xTest[n]*np.sin(thetaCur)**2 - yTest[n]*np.sin(thetaCur)*np.cos(thetaCur))**2 +
                                        (yTest[n]*np.cos(thetaCur)**2 - xTest[n]*np.sin(thetaCur)*np.cos(thetaCur))**2)
                        rCur = xTest[n]*np.cos(thetaCur) + yTest[n]*np.sin(thetaCur)
                        assert np.all(np.diff(rValsInnerCur[-1:half_idx:-1]) > 0) # required for interpolation here
                        zOrigCur = np.interp(rCur, rValsInnerCur[-1:half_idx:-1], zValsInner[-1:half_idx:-1]) # can interpolate because second half has r strictly decreasing (flip order for interp)
                        if dCur < nanopillar_rad + 2*xSteric and zOrigCur < nanopillar_height + 0.2:
                            quad_roots = np.roots([1, -2*(xTest[n]*np.cos(thetaCur)+yTest[n]*np.sin(thetaCur)), 
                                              xTest[n]**2 + yTest[n]**2 - (nanopillar_rad+2*xSteric)**2])
                            if not np.all(np.isreal(quad_roots)):
                                raise ValueError("These roots must be real-valued!")
                            if np.sum(quad_roots < 0)==1: # if r is negative, this should correspond to central nanopillar
                                assert np.isclose(np.abs(quad_roots[0]), np.abs(quad_roots[1])) # satisfied for central nanopillar
                            elif np.sum(quad_roots < 0)==2: # then on the other side
                                continue #raise ValueError("These roots must be positive!")
                            if np.all(quad_roots > rValsInnerCur[half_idx+1]):
                                continue
                            else:
                                rCross.append(quad_roots[0])
                                rCross.append(quad_roots[1])
                    rCross = np.array(rCross)
                    rCross = np.sort(rCross)[-1::-1]
                    sCross = np.zeros_like(rCross)
                    # if not np.any(rCross < 0):
                    #     print("Center pillar should be included, what...?")
                    if len(rCross) == 0:
                        # then no intersections with nanopillars for this value of theta
                        num_lower = np.ceil((sValsInnerCur[-1]-sValsInnerCur[half_idx])/dsBig)
                        sLower = np.linspace(sValsInner[half_idx] + dsBig, sValsInnerCur[-1], int(num_lower))
                        sValsCur = np.concatenate((sValsCur, sLower))
                    for i in range(len(rCross)):
                        if rCross[i] > rValsInnerCur[half_idx+1]: # first intersection occurs in middle of NP
                            continue
                        elif rCross[i] < 0: # central nanopillar
                            continue
                        else:
                            sCross[i] = np.interp(rCross[i], rValsInnerCur[-1:half_idx:-1], sValsInnerCur[-1:half_idx:-1])
                    for i in range(0,len(rCross),2):
                        # define ranges of arc length over each nanopillar crossing region
                        if rCross[i] > rValsInnerCur[half_idx+1]: # first intersection occurs in middle of NP
                            assert np.all(np.diff(zValsInnerCur[-2:0:-1]) > 0) # exlude first and last entry for interpolation
                            sFirstCross = np.interp(nanopillar_height+0.2, zValsInnerCur[-2:0:-1], sValsInnerCur[-2:0:-1])
                            rFirstCross = np.interp(nanopillar_height+0.2, zValsInnerCur[-2:0:-1], rValsInnerCur[-2:0:-1])
                            numBefore = np.floor(2*(sFirstCross - sValsCur[-1])/(dsSmall+dsBig))
                            dsSumCur = 2*(sFirstCross - sValsCur[-1])/numBefore
                            dsBigCur = dsSumCur - dsSmall
                            delta_s = (dsBig-dsSmall)/(numBefore-1)
                        else:
                            numBefore = np.floor(2*(sCross[i] - sValsCur[-1])/(dsSmall+dsBig))
                            dsSumCur = 2*(sCross[i] - sValsCur[-1])/numBefore
                            dsBigCur = dsSumCur - dsSmall
                            delta_s = (dsBigCur-dsSmall)/(numBefore-1)
                            rFirstCross = rCross[i]
                            sFirstCross = sCross[i]
                        if numBefore > 1:
                            sBefore = np.zeros([int(numBefore),])
                            sBefore[0] = sValsCur[-1] + dsBigCur
                            for n in range(1,int(numBefore)):
                                sBefore[n] = sBefore[n-1] + (dsBigCur-n*delta_s)
                        else:
                            sBefore = np.array([sFirstCross])
                        
                        if rCross[i+1] < 0:
                            numIn = np.ceil(rFirstCross/dsSmall)
                            sIn = np.linspace(sFirstCross+dsSmall, sValsInnerCur[-1], int(numIn))
                            sAfter = np.array([])
                        else:
                            numIn = np.ceil((rFirstCross-rCross[i+1])/dsSmall)
                            sIn = np.linspace(sFirstCross+dsSmall, sCross[i+1], int(numIn))
                            if len(rCross) < i+3: #then no central nanopillar!
                                sMidAfter = sValsInnerCur[-1]
                            else:
                                sMidAfter = (sCross[i+1] + sCross[i+2])/2
                            numAfter = np.floor(2*(sMidAfter - sCross[i+1])/(dsSmall+dsBig))
                            dsSumCur = 2*(sMidAfter - sCross[i+1])/numAfter
                            dsBigCur = dsSumCur - dsSmall
                            delta_s = (dsBig-dsSmall)/(numAfter-1)
                            if numAfter > 1:
                                sAfter = np.zeros([int(numAfter),])
                                sAfter[0] = sCross[i+1] + (dsSmall)
                                for n in range(1,int(numAfter)):
                                    sAfter[n] = sAfter[n-1] + (dsSmall+n*delta_s)
                            else:
                                sAfter = np.array([sMidAfter])
                        sValsCur = np.concatenate((sValsCur, sBefore, sIn, sAfter))

                    # interpolate r and z values at specified arc length values
                    rValsCur = np.interp(sValsCur, sValsInnerCur, rValsInnerCur)
                    zValsCur = np.interp(sValsCur, sValsInnerCur, zValsInnerCur)
                    xSteric = 0.05
                    # nanopillar_radmod = nanopillar_rad + 2*xSteric # to avoid collision with PM
                    # define nanopillar locations
                    profileList.append([list(rValsCur), list(zValsCur)])
                    np.savetxt(f"theta{j}.txt", np.array([rValsCur, zValsCur]))
                    for i in range(len(rValsCur)):
                        if i == 0:
                            inner_tag_list.append(top_point)
                        elif i == len(rValsCur)-1:
                            inner_tag_list.append(bottom_point)
                        else:
                            xCur = rValsCur[i]*np.cos(thetaCur)
                            yCur = rValsCur[i]*np.sin(thetaCur)
                            zCur = zValsCur[i]
                            # if u_nuc == []:
                            #     uCur = [0, 0, 0]
                            # else:
                            #     uCur = u_nuc(xRef/rScale, yRef/rScale, zRef-z0Inner)
                            # xCur = xRef# + uCur[0]
                            # yCur = yRef# + uCur[1]
                            # zCur = zRef + uCur[2]
                            cur_tag = gmsh.model.occ.add_point(xCur, yCur, zCur)
                            inner_tag_list.append(cur_tag)
                        # if i > 0 and (j==0 or j==len(thetaVecInner)-1):
                        #     cur_line = gmsh.model.occ.add_line(inner_tag_list[-1], inner_tag_list[-2])
                        #     inner_line_list.append(cur_line)
                    # line_list_list.append(inner_line_list)
                    # if j==0 or j==len(thetaVecInner)-1:
                    #     inner_spline_list.append(inner_line_list)
                    # else:
                    inner_spline_list.append(gmsh.model.occ.add_bspline(inner_tag_list))
                    # first_points_list.append(inner_tag_list[0])
                if j > 0:
                    # if j==1:
                    #     edge_loop_tag = gmsh.model.occ.add_curve_loop(
                    #         [inner_spline_list[j], *inner_spline_list[j-1]])
                    #     edge_surf_list.append(gmsh.model.occ.add_surface_filling(edge_loop_tag))
                    # elif j==len(thetaVecInner)-1:
                    #     edge_loop_tag = gmsh.model.occ.add_curve_loop(
                    #         [inner_spline_list[j-1], *inner_spline_list[j]])
                    #     edge_surf_list.append(gmsh.model.occ.add_surface_filling(edge_loop_tag))
                    # else:
                    # connecting_line = gmsh.model.occ.add_line(first_points_list[-1], first_points_list[-2])
                    edge_loop_tag = gmsh.model.occ.add_curve_loop(
                        [inner_spline_list[j], inner_spline_list[j-1]])
                    cur_surf = gmsh.model.occ.add_bspline_filling(edge_loop_tag, type="Coons")
                    edge_surf_list.append(cur_surf)
                    # xMid = max(rValsInner)*np.cos((thetaVecInner[j]+thetaVecInner[j-1])/2)
                    # yMid = max(rValsInner)*np.sin((thetaVecInner[j]+thetaVecInner[j-1])/2)
                    # substrate_point = gmsh.model.occ.add_point(xMid, yMid, 0.0)
                    # origin_tag = gmsh.model.occ.add_point(0.0, 0.0, 0.0)
                    # front_line = gmsh.model.occ.add_line(substrate_point, first_points_list[-2]) 
                    # back_line = gmsh.model.occ.add_line(substrate_point, first_points_list[-1])
                    # symm_line = gmsh.model.occ.add_line(bottom_point, origin_tag)
                    # substrate_line = gmsh.model.occ.add_line(origin_tag, substrate_point)
                    # front_loop = gmsh.model.occ.add_curve_loop([front_line, inner_spline_list[j-1], symm_line, substrate_line])
                    # back_loop = gmsh.model.occ.add_curve_loop([back_line, inner_spline_list[j], symm_line, substrate_line])
                    # side_loop = gmsh.model.occ.add_curve_loop([connecting_line, front_line, back_line])
                    # front_surf = gmsh.model.occ.add_plane_surface([front_loop])
                    # back_surf = gmsh.model.occ.add_plane_surface([back_loop])
                    # side_surf = gmsh.model.occ.add_plane_surface([side_loop])
                    # cur_surf_loop = gmsh.model.occ.add_surface_loop([front_surf, back_surf, cur_surf, side_surf])
                    # cut_vol_list.append(gmsh.model.add_volume([cur_surf_loop]))
                    
            # for vol in cut_vol_list:
            #     (inner_shape, inner_shape_map) = gmsh.model.occ.cut(inner_shape, [(2, vol)])
            #     inner_shape_list = []
            #     for j in range(len(inner_shape_map)):
            #         if inner_shape_map[j]!=[]:
            #             inner_shape_list.append(inner_shape_map[j][0])
            #     inner_shape = inner_shape_list
            
            if sym_fraction < 1:
                symm_line = gmsh.model.occ.add_line(bottom_point, top_point)
                front_loop = gmsh.model.occ.add_curve_loop([symm_line, inner_spline_list[0]])
                # front_loop = gmsh.model.occ.add_curve_loop([symm_line, *line_list_list[0]])
                front_surf = gmsh.model.occ.add_plane_surface([front_loop])
                back_loop = gmsh.model.occ.add_curve_loop([symm_line, inner_spline_list[-1]])
                # back_loop = gmsh.model.occ.add_curve_loop([symm_line, *line_list_list[-1]])
                back_surf = gmsh.model.occ.add_plane_surface([back_loop])
                inner_surf_loop = gmsh.model.occ.add_surface_loop([front_surf, back_surf, *edge_surf_list])
                inner_shape = gmsh.model.occ.add_volume([inner_surf_loop])
                inner_shape = [(3, inner_shape)]
            else:
                # now define total inner shape from edge_segments and edge_surf_list    
                inner_surf_loop = gmsh.model.occ.add_surface_loop(edge_surf_list)
                inner_shape = gmsh.model.occ.add_volume([inner_surf_loop])
                inner_shape = [(3, inner_shape)]
        else:
            # Inner shape is just a spheroid
            inner_tag_list = []
            for i in range(len(rValsInner)):
                cur_tag = gmsh.model.occ.add_point(rValsInner[i], 0, zValsInner[i])
                inner_tag_list.append(cur_tag)
            inner_spline_tag = gmsh.model.occ.add_spline(inner_tag_list)
            symm_inner_tag = gmsh.model.occ.add_line(inner_tag_list[0], inner_tag_list[-1])
            inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, symm_inner_tag])
            inner_plane_tag = gmsh.model.occ.add_plane_surface([inner_loop_tag])
            inner_shape = gmsh.model.occ.revolve([(2, inner_plane_tag)], 
                                                0, 0, 0, 0, 0, 1, 2 * np.pi * sym_fraction)

        inner_shape_tags = []
        for i in range(len(inner_shape)):
            if inner_shape[i][0] == 3:  # pull out tags associated with 3d objects
                inner_shape_tags.append(inner_shape[i][1])
        assert len(inner_shape_tags) == 1  # should be just one 3D body from the full revolution

        # Create interface between 2 objects
        two_shapes, (outer_shape_map, inner_shape_map) = gmsh.model.occ.fragment(
            [(3, outer_shape_tags[0])], [(3, inner_shape_tags[0])]
        )
        gmsh.model.occ.synchronize()

        # Get the outer boundary
        outer_shell = gmsh.model.getBoundary(two_shapes, oriented=False)
        outer_shell_tags = []
        for i in range(len(outer_shell)):
            outer_shell_tags.append(outer_shell[i][1])
        # assert (
        #     len(outer_shell) == 2
        # )  # 2 boundaries because of bottom surface at z = 0, both belong to PM
        # Get the inner boundary
        inner_shell = gmsh.model.getBoundary(inner_shape_map, oriented=False)
        inner_shell_tags = []
        for i in range(len(inner_shell)):
            inner_shell_tags.append(inner_shell[i][1])
        # assert len(inner_shell) == 1
        # Add physical markers for facets
        gmsh.model.add_physical_group(
            outer_shell[0][0], outer_shell_tags, tag=outer_marker
        )
        gmsh.model.add_physical_group(inner_shell[0][0], inner_shell_tags, tag=interface_marker)
        # if nuc_compression > 0:
        #     for i in range(len(inner_shell_tags)):
        #         gmsh.model.mesh.setSmoothing(2, inner_shell_tags[i], 2)

        # Physical markers for
        all_volumes = [tag[1] for tag in outer_shape_map]
        inner_volume = [tag[1] for tag in inner_shape_map]
        outer_volume = []
        for vol in all_volumes:
            if vol not in inner_volume:
                outer_volume.append(vol)
        gmsh.model.add_physical_group(3, outer_volume, tag=outer_vol_tag)
        gmsh.model.add_physical_group(3, inner_volume, tag=inner_vol_tag)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM and hInnerEdge at the inner membrane
        # between these, the value is interpolated based on the relative distance
        # between the two membranes.
        # Inside the inner shape, the value is interpolated between hInnerEdge
        # and lc3, where lc3 = max(hInnerEdge, 0.2*maxInnerDim)
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*maxOuterDim in the center
        rCur = np.sqrt(x**2 + y**2)
        RCur = np.sqrt(rCur**2 + (z - zMid) ** 2)
        outer_dist = np.sqrt((rCur - rValsOuter) ** 2 + (z - zValsOuter) ** 2)
        np.append(outer_dist, z)  # include the distance from the substrate
        if np.all(np.array(nanopillars) != 0):
            xy_NP_dist = min(np.sqrt((x-xTest)**2 + (y-yTest)**2))
            if xy_NP_dist < nanopillar_rad:
                xy_NP_dist = 0
            else:
                xy_NP_dist -= nanopillar_rad
            z_NP_dist = max([0, z - nanopillar_height])
            dist_to_NP = np.sqrt(xy_NP_dist**2 + z_NP_dist**2)
            hNPWeight = np.exp(-dist_to_NP / 2.5)
        else:
            hNPWeight = 0
        dist_to_outer = min(outer_dist)
        if len(rValsInner) == 0:
            lc3 = 0.2 * maxOuterDim
            dist_to_inner = RCur
            in_outer = True
        else:
            inner_dist = np.sqrt((rCur - rValsInner) ** 2 + (z - zValsInner) ** 2)
            dist_to_inner = min(inner_dist)
            inner_idx = np.argmin(inner_dist)
            inner_rad = RInnerVec[inner_idx]
            R_rel_inner = RCur / inner_rad
            lc3 = max(hInnerEdge, 0.2 * maxInnerDim)
            in_outer = R_rel_inner > 1
        lc1 = (1-hNPWeight)*hEdge + hNPWeight*hNP
        lc2 = (1-hNPWeight)*hInnerEdge + hNPWeight*hNP
        lc3 = (1-hNPWeight)*lc3 + hNPWeight*hNP
        if in_outer:
            lcTest = lc1 + (lc2 - lc1) * (dist_to_outer) / (dist_to_inner + dist_to_outer)
        else:
            lcTest = lc2 + (lc3 - lc2) * (1 - R_rel_inner)
        return lcTest

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # which may provide better results when there are larger gradients in mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(3)
    rank = MPI.COMM_WORLD.rank
    if use_tmp:
        tmp_folder = pathlib.Path(f"/root/tmp/tmp_3dcell_{rank}")
    else:
        tmp_folder = pathlib.Path(f"tmp_3dcell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "3dcell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # cur_map = d.vertex_to_dof_map(u_nuc.function_space())
    # areas = d.MeshFunction("double", mesh_bound, 2, 1)
    # for c in d.cells(mesh_bound):
    #     area_orig = c.volume()
    #     coords = np.array(c.get_vertex_coordinates())
    #     u_cur = []
    #     for v in d.vertices(c):
    #         u_cur.append([u_nuc.vector()[cur_map[3*v.index()]:cur_map[3*v.index()]+3]])
    #     coords += np.array(u_cur).flatten()
    #     area_vec = 0.5*np.cross(coords[0:3]-coords[3:6], coords[3:6]-coords[6:9])
    #     areas[c.index()] = np.sqrt(area_vec[0]**2 + area_vec[1]**2 + area_vec[2]**2) / area_orig
    
    if nuc_compression > 0:
        a_nuc = compute_stretch(u_nuc, aInner, bInner, nanopillars, z0Inner)
        d.ALE.move(u_nuc.function_space().mesh(), u_nuc)
    else:
        a_nuc = []

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags) and substrate_markers (2d)
    substrate_markers = d.MeshFunction("size_t", dmesh, 2)
    for f in d.facets(dmesh):
        topology, cellIndices = facet_topology(f, mf3)
        if topology == "boundary":
            # test if it is on the outward surface (not substrate)
            rCur = np.sqrt(f.midpoint().x()**2 + f.midpoint().y()**2)
            zCur = f.midpoint().z()
            thetaCur = np.arctan2(f.midpoint().y(), f.midpoint().x())
            if zCur == 0:
                mf2.set_value(f.index(), outer_marker)
                substrate_markers.set_value(f.index(), outer_marker)
            elif sym_fraction != 1 and (
                np.isclose(thetaCur, 0.0, atol=1e-4) or np.isclose(thetaCur, 2*np.pi*sym_fraction, atol=1e-4)):
                # then a no flux surface
                mf2.set_value(f.index(), 0)
            else: #then either on a nanopillar or outer surface
                mf2.set_value(f.index(), outer_marker)
                if np.all(np.array(nanopillars) != 0):
                    if rCur < contactRad - xCurv + hEdge/100 and zCur < zOffset + hEdge/100:
                        substrate_markers.set_value(f.index(), outer_marker)

    if return_curvature:
        curv_markers = d.MeshFunction("double", dmesh, 0)#compute_curvature(dmesh, mf2, mf3, facet_list, cell_list)
        # first set curvatures on substrate
        for f in d.facets(dmesh):
            if substrate_markers.array()[f.index()] == outer_marker:
                for v in d.vertices(f):
                    # explictly set curvature for membrane on nanopillars (or set to zero for no nanopillars)
                    zVal = v.midpoint().z()
                    if zVal >= nanopillar_height:
                        curv_markers.set_value(v.index(), 0.0)
                    elif zVal > nanopillar_height - xSteric: # curved from cyl to top
                        cosCur = (zVal - (nanopillar_height-xSteric))/xSteric
                        sinCur = np.sqrt(1 - cosCur**2)
                        cm = 1/xSteric
                        cp = sinCur / (nanopillar_rad + xSteric*sinCur)
                        rVal = nanopillar_rad + xSteric*sinCur
                        cpAlt = (rVal - nanopillar_rad) / (rVal*xSteric)
                        curv_markers.set_value(v.index(), -(cm+cp)/2) 
                    elif zVal > xCurv: # cylinder
                        curv_markers.set_value(v.index(), -0.5/nanopillar_rad)
                    elif zVal > 0: # curved to substrate
                        cosCur = (xCurv - zVal)/xCurv
                        sinCur = np.sqrt(1 - cosCur**2)
                        cm = 1/xCurv
                        cp = -sinCur / (nanopillar_rad + xSteric + xCurv*(1-sinCur))
                        rVal = nanopillar_rad + xSteric + xCurv - xCurv*sinCur
                        cpAlt = (rVal - (nanopillar_rad + xSteric + xCurv)) / (rVal*xCurv)
                        curv_markers.set_value(v.index(), (cm+cp)/2)
                    else: # substrate
                        curv_markers.set_value(v.index(), 0.0)
        
        # curvature on PM and NM
        for f in d.facets(dmesh):
            if (substrate_markers.array()[f.index()] == 0 and 
                mf2.array()[f.index()] == outer_marker):
                # set curvature at other boundaries (map from 2d mesh case)
                for v in d.vertices(f):
                    rVal = np.sqrt(v.midpoint().x()**2 + v.midpoint().y()**2)
                    zVal = v.midpoint().z()
                    curv_markers.set_value(v.index(), curvFcnOuter(rVal, 0, zVal))
            elif mf2.array()[f.index()] == interface_marker:
                for v in d.vertices(f):
                    xVal = v.midpoint().x()
                    yVal = v.midpoint().y()
                    rVal = np.sqrt(xVal**2 + yVal**2)
                    zVal = v.midpoint().z()
                    if nuc_compression > 0:
                        curv_val = calc_curv_NP(xVal,yVal,zVal,aInner,bInner,z0Inner,xNP,yNP,nanopillars)
                        curv_markers.set_value(v.index(), curv_val)
                    else:
                        curv_val = compute_curvature_ellipse_alt(np.array([rVal]), np.array([zVal]), 
                                                                aInner, bInner, 
                                                                r0Inner, z0Inner, incl_parallel=True)
                        curv_markers.set_value(v.index(), curv_val[0])
        return (dmesh, mf2, mf3, substrate_markers, curv_markers, u_nuc, a_nuc)
    else:
        return (dmesh, mf2, mf3, substrate_markers, u_nuc, a_nuc)

def create_substrate(
    LBox: float = 20.0,
    TBox: float = 1.0,
    hEdge: float = 0,
    outer_marker: int = 10,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
    nanopillars: Tuple[float, float, float] = [0, 0, 0],
    use_tmp: bool = False,
    contact_rad: float = np.inf,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a mesh of the substrate.
    """
    import gmsh
    nanopillar_rad = nanopillars[0]
    nanopillar_height = nanopillars[1]
    nanopillar_spacing = nanopillars[2]
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("3dcell")

    # first add outer body and revolve
    outer_shape = gmsh.model.occ.add_box(-LBox/2,-LBox/2,-TBox,LBox,LBox,TBox)
    outer_shape = [(3, outer_shape)]
    
    if np.all(np.array(nanopillars) != 0):
        num_pillars = 2*np.floor(LBox/(2*nanopillar_spacing)) + 1
        rMax = nanopillar_spacing * np.floor(LBox/(2*nanopillar_spacing))
        test_coords = np.linspace(-rMax, rMax, int(num_pillars))
        xTest, yTest = np.meshgrid(test_coords, test_coords)
        xTest = np.reshape(xTest, (len(xTest)**2))
        yTest = np.reshape(yTest, (len(yTest)**2))
        for i in range(len(xTest)):
            rTest = np.sqrt(xTest[i]**2 + yTest[i]**2)
            xCurv = 0.2
            xSteric = 0.05
            keep_logic1 = rTest <= contact_rad-nanopillar_rad-xCurv-xSteric
            keep_logic2 = rTest > contact_rad+nanopillar_rad
            keep_logic = np.logical_or(keep_logic1, keep_logic2)
            if keep_logic:
                np_shape = gmsh.model.occ.add_cylinder(xTest[i], yTest[i], 0.0, 0, 0, nanopillar_height, nanopillar_rad)
                # np_shape_tags = []
                # for j in range(len(np_shape)):
                #     if np_shape[j][0] == 3:  # pull out tags associated with 3d objects
                #         np_shape_tags.append(np_shape[j][1])
                # assert len(np_shape_tags) == 1  # should be just one 3D body from the full revolution
                (outer_shape, outer_shape_map) = gmsh.model.occ.fuse(outer_shape, [(3, np_shape)])
                outer_shape_list = []
                # for j in range(len(outer_shape_map)):
                #     if outer_shape_map[j]!=[]:
                #         outer_shape_list.append(outer_shape_map[j][0])
                # outer_shape = outer_shape_list

    outer_shape_tags = []
    for i in range(len(outer_shape)):
        if outer_shape[i][0] == 3:  # pull out tags associated with 3d objects
            outer_shape_tags.append(outer_shape[i][1])
    # assert len(outer_shape_tags) == 1  # should be just one 3D body from the full revolution
    gmsh.model.occ.synchronize()
    gmsh.model.add_physical_group(3, outer_shape_tags, tag=outer_vol_tag)
    facets = gmsh.model.getBoundary([(3, outer_shape_tags[0])])
    facet_tags = []
    for i in range(len(facets)):
        facet_tags.append(facets[i][1])
    gmsh.model.add_physical_group(2, facet_tags, tag=outer_marker)

    # def meshSizeCallback(dim, tag, x, y, z, lc):
    #     return hEdge

    # gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # # set off the other options for mesh size determination
    # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # # which may provide better results when there are larger gradients in mesh size
    # gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(3)
    rank = MPI.COMM_WORLD.rank
    if use_tmp:
        tmp_folder = pathlib.Path(f"/root/tmp/tmp_3dcell_{rank}")
    else:
        tmp_folder = pathlib.Path(f"tmp_3dcell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "substrate.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    return dmesh, mf2, mf3


def create_2Dcell(
    contactRad: float = 13.0,
    hEdge: float = 0,
    hInnerEdge: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_tag: int = 2,
    outer_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
    half_cell: bool = True,
    return_curvature: bool = False,
    axisymm: bool = True,
    use_tmp: bool = False,
    nanopillar: bool = False,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a 2D mesh of a cell profile, with the bounding curve defined in
    terms of r and z (e.g. unit circle would be "r**2 + (z-1)**2 == 1)
    It is assumed that substrate is present at z = 0, so if the curve extends
    below z = 0 , there is a sharp cutoff.
    If half_cell = True, only have of the contour is constructed, with a
    left zero-flux boundary at r = 0.
    Can include one compartment inside another compartment.
    Recommended for use with the axisymmetric feature of SMART.

    Args:
        outerExpr: String implicitly defining an r-z curve for the outer surface
        innerExpr: String implicitly defining an r-z curve for the inner surface
        hEdge: maximum mesh size at the outer edge
        hInnerEdge: maximum mesh size at the edge
            of the inner compartment
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on edge of the outer ellipse with
        inner_tag: The value to mark the inner ellipse surface with
        outer_tag: The value to mark the outer ellipse surface with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
        half_cell: If true, consider r=0 the symmetry axis for an axisymm shape
        use_tmp: argument to use tmp directory for gmsh file creation
        nanopillar: if true, add a central nanopillar 
    Returns:
        Tuple (mesh, facet_marker, cell_marker)
    """
    import gmsh

    aVecRef = np.array([35, 2000, 1000, 1200, 1500, 1000])
    bVecRef = np.array([0.94, 0.4, 0.4, 0.4, 0.4, 0.4])
    cVecRef = np.array([0.01, 9.72, 15, 24.5, 33.2, 42])
    dVecRef = np.array([30, 15, 10, 1.0, 0.12, 0.0322])
    # innerExprList = [innerExpr10, innerExpr13, innerExpr15, innerExpr25]
    RList = np.array([10, 13, 15, 20, 25, 30])
    scaleFactor = 0.82
    nucScaleFactor = 0.8
    sym_fraction = 1/8
    aVecRef = aVecRef / scaleFactor**4
    cVecRef = cVecRef / scaleFactor
    dVecRef = dVecRef / scaleFactor**4
    RList = RList / scaleFactor
    findMatch = np.isclose(RList, contactRad)
    if np.any(findMatch):
        idx = np.nonzero(findMatch)[0][0]
        refParam = [aVecRef[idx], bVecRef[idx], cVecRef[idx], dVecRef[idx], RList[idx]]
    elif contactRad < max(RList) and contactRad > min(RList):
        aVal = np.interp(contactRad, np.array(RList), np.array(aVecRef))
        bVal = np.interp(contactRad, np.array(RList), np.array(bVecRef))
        cVal = np.interp(contactRad, np.array(RList), np.array(cVecRef))
        dVal = np.interp(contactRad, np.array(RList), np.array(dVecRef))
        refParam = [aVal, bVal, cVal, dVal, contactRad]
    else:
        raise ValueError("This shape is outside the specified range")

    nuc_vol = (4/3)*np.pi*5.3*5.3*2.4/nucScaleFactor**3
    zOffset = 0
    targetVol = 480/scaleFactor**3 *4 + nuc_vol
    rValsOuter, zValsOuter, outParam = shape_adj_axisymm(refParam, zOffset, targetVol)
    rValsOuter, zValsOuter = dilate_axisymm(rValsOuter, zValsOuter, targetVol)
    zMax = max(zValsOuter)

    innerExpr = get_inner(zOffset, zMax, nucScaleFactor, 0.0)

    if return_curvature:
        # create full mesh for curvature analysis and then map onto half mesh
        # if half_cell_with_curvature is True
        half_cell_with_curvature = half_cell
        half_cell = False

    if not len(rValsInner) == 0:
        rValsInner, zValsInner = implicit_curve(innerExpr)
        zMid = np.mean(zValsInner)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        RInnerVec = np.sqrt(rValsInner**2 + (zValsInner - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
        maxInnerDim = max(RInnerVec)
    else:
        zMid = np.mean(zValsOuter)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * maxOuterDim
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.2 * maxOuterDim if len(rValsInner) == 0 else 0.2 * maxInnerDim
    # Create the 2D mesh using gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("2DCell")
    # first add outer body and revolve
    outer_tag_list = []
    for i in range(len(rValsOuter)):
        cur_tag = gmsh.model.occ.add_point(rValsOuter[i], 0, zValsOuter[i])
        outer_tag_list.append(cur_tag)
    outer_spline_tag = gmsh.model.occ.add_spline(outer_tag_list)
    if not half_cell:
        outer_tag_list2 = []
        for i in range(len(rValsOuter)):
            cur_tag = gmsh.model.occ.add_point(-rValsOuter[i], 0, zValsOuter[i])
            outer_tag_list2.append(cur_tag)
        outer_spline_tag2 = gmsh.model.occ.add_spline(outer_tag_list2)
    if np.isclose(zValsOuter[-1], 0):  # then include substrate at z=0
        if half_cell:
            if nanopillar:
                contact_tag = gmsh.model.occ.add_point(0.3, 0, 0)
                nanopillar_tag_list = [contact_tag]
                xCur = 0.3
                zCur = 0.0
                thetaCur = 0.0
                while xCur > 0:
                    if zCur < 3.0:
                        zCur += 0.01
                    else:
                        thetaCur += np.pi/20
                        zCur = 3.0 + 0.3*np.sin(thetaCur)
                        xCur = 0.3*np.cos(thetaCur)
                    cur_tag = gmsh.model.occ.add_point(xCur, 0, zCur)
                    nanopillar_tag_list.append(cur_tag)
                left_tag = gmsh.model.occ.add_point(0, 0, 3.3)
                nanopillar_tag_list.append(left_tag)
                nanopillar_spline_tag = gmsh.model.occ.add_spline(nanopillar_tag_list)
                symm_axis_tag = gmsh.model.occ.add_line(left_tag, outer_tag_list[0])
                bottom_tag = gmsh.model.occ.add_line(contact_tag, outer_tag_list[-1])
                outer_loop_tag = gmsh.model.occ.add_curve_loop(
                    [outer_spline_tag, bottom_tag, nanopillar_spline_tag, symm_axis_tag]
                )
            else:
                origin_tag = gmsh.model.occ.add_point(0, 0, 0)
                symm_axis_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[0])
                bottom_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[-1])
                outer_loop_tag = gmsh.model.occ.add_curve_loop(
                    [outer_spline_tag, bottom_tag, symm_axis_tag]
                )
        else:
            bottom_tag = gmsh.model.occ.add_line(outer_tag_list[-1], outer_tag_list2[-1])
            outer_loop_tag = gmsh.model.occ.add_curve_loop(
                [outer_spline_tag, outer_spline_tag2, bottom_tag]
            )
    else:
        if half_cell:
            symm_axis_tag = gmsh.model.occ.add_line(outer_tag_list[0], outer_tag_list[-1])
            outer_loop_tag = gmsh.model.occ.add_curve_loop([outer_spline_tag, symm_axis_tag])
        else:
            outer_loop_tag = gmsh.model.occ.add_curve_loop([outer_spline_tag, outer_spline_tag2])
    cell_plane_tag = gmsh.model.occ.add_plane_surface([outer_loop_tag])

    if innerExpr == "":
        # No inner shape in this case
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(2, [cell_plane_tag], tag=outer_tag)
        facets = gmsh.model.getBoundary([(2, cell_plane_tag)])
        facet_tag_list = []
        for i in range(len(facets)):
            facet_tag_list.append(facets[i][1])
        if half_cell:  # if half, set symmetry axis to 0 (no flux)
            xmin, ymin, zmin = (-hInnerEdge / 10, -hInnerEdge / 10, -1)
            xmax, ymax, zmax = (hInnerEdge / 10, hInnerEdge / 10, max(zValsOuter) + 1)
            all_symm_bound = gmsh.model.occ.get_entities_in_bounding_box(
                xmin, ymin, zmin, xmax, ymax, zmax, dim=1
            )
            symm_bound_markers = []
            for i in range(len(all_symm_bound)):
                symm_bound_markers.append(all_symm_bound[i][1])
            gmsh.model.add_physical_group(1, symm_bound_markers, tag=0)
        gmsh.model.add_physical_group(1, facet_tag_list, tag=outer_marker)
    else:
        # Add inner shape
        inner_tag_list = []
        if nanopillar:
            logical_idx = np.logical_or(rValsInner > 0.4, zValsInner > 3.4)
            rValsInner = rValsInner[logical_idx]
            zValsInner = zValsInner[logical_idx]
            rValsInner = np.append(rValsInner, 0.4)
            zValsInner = np.append(zValsInner, zValsInner[-1])
        for i in range(len(rValsInner)):
            cur_tag = gmsh.model.occ.add_point(rValsInner[i], 0, zValsInner[i])
            inner_tag_list.append(cur_tag)
        inner_spline_tag = gmsh.model.occ.add_spline(inner_tag_list)
        if half_cell:
            if nanopillar:
                contact_tag = gmsh.model.occ.add_point(0.4, 0, zValsInner[-1])
                nanopillar_tag_list = [contact_tag]
                xCur = 0.4
                zCur = zValsInner[-1]
                thetaCur = 0.0
                while xCur > 0:
                    if zCur < 3.0:
                        zCur += 0.01
                    else:
                        thetaCur += np.pi/20
                        zCur = 3.0 + 0.4*np.sin(thetaCur)
                        xCur = 0.4*np.cos(thetaCur)
                    cur_tag = gmsh.model.occ.add_point(xCur, 0, zCur)
                    nanopillar_tag_list.append(cur_tag)
                left_tag = gmsh.model.occ.add_point(0, 0, 3.4)
                nanopillar_tag_list.append(left_tag)
                nanopillar_spline_tag_inner = gmsh.model.occ.add_spline(nanopillar_tag_list)
                symm_inner_tag = gmsh.model.occ.add_line(left_tag, inner_tag_list[0])
                inner_loop_tag = gmsh.model.occ.add_curve_loop(
                    [inner_spline_tag, nanopillar_spline_tag_inner, symm_inner_tag]
                )
            else:
                symm_inner_tag = gmsh.model.occ.add_line(inner_tag_list[0], inner_tag_list[-1])
                inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, symm_inner_tag])
        else:
            inner_tag_list2 = []
            for i in range(len(rValsInner)):
                cur_tag = gmsh.model.occ.add_point(-rValsInner[i], 0, zValsInner[i])
                inner_tag_list2.append(cur_tag)
            inner_spline_tag2 = gmsh.model.occ.add_spline(inner_tag_list2)
            inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, inner_spline_tag2])
        inner_plane_tag = gmsh.model.occ.add_plane_surface([inner_loop_tag])
        cell_plane_list = [cell_plane_tag]
        inner_plane_list = [inner_plane_tag]

        outer_volume = []
        inner_volume = []
        all_volumes = []
        inner_marker_list = []
        outer_marker_list = []
        for i in range(len(cell_plane_list)):
            cell_plane_tag = cell_plane_list[i]
            inner_plane_tag = inner_plane_list[i]
            # Create interface between 2 objects
            two_shapes, (outer_shape_map, inner_shape_map) = gmsh.model.occ.fragment(
                [(2, cell_plane_tag)], [(2, inner_plane_tag)]
            )
            gmsh.model.occ.synchronize()

            # Get the outer boundary
            outer_shell = gmsh.model.getBoundary(two_shapes, oriented=False)
            for i in range(len(outer_shell)):
                outer_marker_list.append(outer_shell[i][1])
            # Get the inner boundary
            inner_shell = gmsh.model.getBoundary(inner_shape_map, oriented=False)
            for i in range(len(inner_shell)):
                inner_marker_list.append(inner_shell[i][1])
            for tag in outer_shape_map:
                all_volumes.append(tag[1])
            for tag in inner_shape_map:
                inner_volume.append(tag[1])

            for vol in all_volumes:
                if vol not in inner_volume:
                    outer_volume.append(vol)

        # Add physical markers for facets
        if half_cell:  # if half, set symmetry axis to 0 (no flux)
            xmin, ymin, zmin = (-hInnerEdge / 10, -hInnerEdge / 10, -1)
            xmax, ymax, zmax = (hInnerEdge / 10, hInnerEdge / 10, max(zValsOuter) + 1)
            all_symm_bound = gmsh.model.occ.get_entities_in_bounding_box(
                xmin, ymin, zmin, xmax, ymax, zmax, dim=1
            )
            symm_bound_markers = []
            for i in range(len(all_symm_bound)):
                symm_bound_markers.append(all_symm_bound[i][1])
            # note that this first call sets the symmetry axis to tag 0 and
            # this is not overwritten by the next calls to add_physical_group
            gmsh.model.add_physical_group(1, symm_bound_markers, tag=0)
        gmsh.model.add_physical_group(1, outer_marker_list, tag=outer_marker)
        gmsh.model.add_physical_group(1, inner_marker_list, tag=interface_marker)

        # Physical markers for "volumes"
        gmsh.model.add_physical_group(2, outer_volume, tag=outer_tag)
        gmsh.model.add_physical_group(2, inner_volume, tag=inner_tag)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM and hInnerEdge at the inner membrane
        # between these, the value is interpolated based on the relative distance
        # between the two membranes.
        # Inside the inner shape, the value is interpolated between hInnerEdge
        # and lc3, where lc3 = max(hInnerEdge, 0.2*maxInnerDim)
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*maxOuterDim in the center
        rCur = np.sqrt(x**2 + y**2)
        RCur = np.sqrt(rCur**2 + (z - zMid) ** 2)
        outer_dist = np.sqrt((rCur - rValsOuter) ** 2 + (z - zValsOuter) ** 2)
        np.append(outer_dist, z)  # include the distance from the substrate
        dist_to_outer = min(outer_dist)
        if innerExpr == "":
            lc3 = 0.2 * maxOuterDim
            dist_to_inner = RCur
            in_outer = True
        else:
            inner_dist = np.sqrt((rCur - rValsInner) ** 2 + (z - zValsInner) ** 2)
            dist_to_inner = min(inner_dist)
            inner_idx = np.argmin(inner_dist)
            inner_rad = RInnerVec[inner_idx]
            R_rel_inner = RCur / inner_rad
            lc3 = max(hInnerEdge, 0.2 * maxInnerDim)
            in_outer = R_rel_inner > 1
        lc1 = hEdge
        lc2 = hInnerEdge
        if in_outer:
            lcTest = lc1 + (lc2 - lc1) * (dist_to_outer) / (dist_to_inner + dist_to_outer)
        else:
            lcTest = lc2 + (lc3 - lc2) * (1 - R_rel_inner)
        return lcTest

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # which may provide better results when there are larger gradients in mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(2)
    rank = MPI.COMM_WORLD.rank
    if use_tmp:
        tmp_folder = pathlib.Path(f"/root/tmp/tmp_2DCell_{rank}")
    else:
        tmp_folder = pathlib.Path(f"tmp_2DCell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "2DCell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 2, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags)
    if return_curvature:
        if innerExpr == "":
            facet_list = [outer_marker]
            cell_list = [outer_tag]
        else:
            facet_list = [outer_marker, interface_marker]
            cell_list = [outer_tag, inner_tag]
        if half_cell_with_curvature:  # will likely not work in parallel...
            dmesh_half, mf2_half, mf3_half = create_2Dcell(
                contactRad,
                hEdge,
                hInnerEdge,
                interface_marker,
                outer_marker,
                inner_tag,
                outer_tag,
                comm,
                verbose,
                half_cell=True,
                return_curvature=False,
            )
            kappa_mf = compute_curvature(
                dmesh, mf2, mf3, facet_list, cell_list, half_mesh_data=(dmesh_half, mf2_half),
                axisymm=axisymm,
            )
            (dmesh, mf2, mf3) = (dmesh_half, mf2_half, mf3_half)
        else:
            kappa_mf = compute_curvature(dmesh, mf2, mf3, facet_list, cell_list, axisymm=axisymm)
        return (dmesh, mf2, mf3, kappa_mf)
    else:
        return (dmesh, mf2, mf3)
    
def calc_vol_axisymm(rVec, zVec):
    vol = 0
    for i in range(1, len(rVec)):
        dr = rVec[i-1] - rVec[i]
        dz = zVec[i-1] - zVec[i]
        vol += np.pi*(rVec[i]**2 * dz + rVec[i]*dr*dz + dr**2 * dz / 3)
    return vol

def dilate_axisymm(rVec, zVec, volTarget, zThresh):
    volCur = calc_vol_axisymm(rVec, zVec)
    normals = [[0, 1]]
    sTot = 0
    for i in range(1, len(rVec)-1):
        dr1 = rVec[i] - rVec[i-1]
        dz1 = zVec[i] - zVec[i-1]
        ds1 = np.sqrt(dr1**2 + dz1**2)
        dr2 = rVec[i+1] - rVec[i]
        dz2 = zVec[i+1] - zVec[i]
        ds2 = np.sqrt(dr2**2 + dz2**2)
        curVec = [-(dz1/ds1+dz2/ds2)/2, (dr1/ds1+dr2/ds2)/2]
        curVec[0] = curVec[0] / np.sqrt(curVec[0]**2 + curVec[1]**2)
        curVec[1] = curVec[1] / np.sqrt(curVec[0]**2 + curVec[1]**2)
        normals.append(curVec)
        sTot += np.sqrt(ds1)
    sTot += np.sqrt(ds2)
    normals.append([1, 0])
    normalMove = 1.0*sTot/len(rVec)
    curSign = np.sign(volTarget-volCur)
    firstIt = True
    volPrev = volCur
    adjIdx = np.nonzero(zVec <= zThresh + 1e-12)
    if len(adjIdx[0]) == 0:
        adjIdx = len(rVec)
    else:
        adjIdx = adjIdx[0][0]
    while np.abs(volCur-volTarget) > 0.001*volTarget:
        for i in range(adjIdx-1):
            curMove = normalMove*np.sign(volTarget-volCur)*(len(rVec)-i)/len(rVec)
            rVec[i] += curMove*normals[i][0]
            zVec[i] += curMove*normals[i][1]
        volCur = calc_vol_axisymm(rVec, zVec)
        if firstIt:
            if curSign == np.sign(volTarget-volCur):
                # raise ValueError("Desired volume is too far off for adjustments")
                return (rVec, zVec)
            else:
                firstIt = False
        normalMove = normalMove * np.abs((volTarget-volCur)/(volCur-volPrev))
        volPrev = volCur
    return (rVec, zVec)

def make_expr(a, b, c, d, R):
        return (f"(1 - z**4/({a}+z**4)) * (r**2 + z**2) + "
                f"{b}*(r**2 + (z+{c})**2)*z**4 / ({d} + z**4) - {R**2}")

def shape_adj_axisymm(paramVec, zOffset, volTarget):
    rVec, zVec = implicit_curve(make_expr(*paramVec))
    zVec[0:-2] += zOffset
    volCur = calc_vol_axisymm(rVec, zVec)
    curSign = np.sign(volTarget-volCur)
    firstIt = True
    volPrev = volCur
    magChange = 50
    while np.abs(volCur-volTarget) > 0.01*volTarget:
        if volCur < volTarget:
            paramVec[0] = paramVec[0]/np.exp(np.log(magChange)/2)
            paramVec[1] = paramVec[1]/np.exp(np.log(magChange)/4)
            paramVec[2] = paramVec[2]/np.exp(np.log(magChange)/4)
            paramVec[3] = paramVec[3]*magChange
        else:
            paramVec[0] = paramVec[0]*np.exp(np.log(magChange)/2)
            paramVec[1] = paramVec[1]*np.exp(np.log(magChange)/4)
            paramVec[2] = paramVec[2]*np.exp(np.log(magChange)/4)
            paramVec[3] = paramVec[3]/magChange
        rVec, zVec = implicit_curve(make_expr(*paramVec))
        zVec[0:-2] += zOffset
        volCur = calc_vol_axisymm(rVec, zVec)
        if firstIt:
            if curSign == np.sign(volTarget-volCur):
                # raise ValueError("Desired volume is too far off for adjustments")
                return (rVec, zVec)
            else:
                firstIt = False
        magChange = 1 + (magChange-1)*np.abs((volTarget-volCur)/(volCur-volPrev))
        volPrev = volCur
        
    return (rVec, zVec, paramVec)

def get_inner(zOffset, zMax, scaleFactor, nuc_compression, aMod=0, bMod=0):
    # zMid = (zOffset + zMax) / 2
    zRad = 2.4/scaleFactor + bMod
    rRad = 5.3/scaleFactor + aMod
    if (zMax - zOffset - 1.0 + nuc_compression) <= zRad*2:
        zRad = (zMax-zOffset-1.0 + nuc_compression)/2 + bMod
        rRad = np.sqrt((5.3**2 * 2.4/scaleFactor**3) / zRad) + aMod
        if zRad < 0:
            raise ValueError("Nucleus does not fit")
    zMid = zOffset + zRad + 0.2 - nuc_compression
    return f"(r/{rRad})**2 + ((z-{zMid})/{zRad})**2 - 1"

def get_inner_param(zOffset, zMax, scaleFactor, nuc_compression, aMod=0, bMod=0):
    zRad = 2.4/scaleFactor + bMod
    rRad = 5.3/scaleFactor + aMod
    if (zMax - zOffset - 1.0 + nuc_compression) <= zRad*2:
        zRad = (zMax-zOffset-1.0 + nuc_compression)/2 + bMod
        rRad = np.sqrt((5.3**2 * 2.4/scaleFactor**3) / zRad) + aMod
        if zRad < 0:
            raise ValueError("Nucleus does not fit")
    zMid = zOffset + zRad + 0.2 - nuc_compression
    return (rRad, zRad, 0.0, zMid)

def compute_curvature_1D(
    rVals, zVals, 
    curvRes: float = 0.1,
    incl_parallel: bool = False, 
    use_tmp = False,
    comm: MPI.Comm = d.MPI.comm_world,
):
    """
    Use dolfin functions to estimate curvature on boundary mesh.
    Boundary meshes are created by extracting the Meshview associated
    with each facet marker value in facet_marker_vec.
    The length of cell_marker_vec must be the same length as facet_marker_vec,
    with each value in the list identifying the marker value for a domain that
    contains the boundary identified by the associated facet_marker.
    For instance, if facet_marker_vec=[10,12] and cell_marker_vec=[1,2], then
    the domain over which mf_cell=1 must contain the domain mf_facet=10 on its boundary
    and the domain mf_cell=2 must contain domain mf_facet=12 on its boundary.
    Mean curvature is approximated by first projecting facet normals (n) onto a boundary
    finite element space and then solving a variational form of kappa = -div(n).

    Args:
        ref_mesh: dolfin mesh describing the entire geometry (parent mesh)
        mf_facet: facet mesh function with boundary domain markers
        mf_cell: cell mesh function with cell domain markers
        facet_marker_vec: list with values of facet markers to iterate over
        cell_marker_vec: list with values of cell markers (see above)
        half_mesh_data: tuple with dolfin mesh for half domain and
                        facet mesh function over half domain. If not specified,
                        this is empty and the curvature values are not mapped
                        onto the half domain
    Returns:
        kappa_mf: Vertex mesh function containing mean curvature values
    """
    facet_marker = 10
    cell_marker = 1
    import gmsh
    gmsh.initialize()
    outer_tag_list = []
    line_list = []
    for i in range(len(rVals)):
        cur_tag = gmsh.model.occ.add_point(rVals[i], 0, zVals[i])
        outer_tag_list.append(cur_tag)
        if i > 0:
            cur_line = gmsh.model.occ.add_line(outer_tag_list[-2], cur_tag)
            line_list.append(cur_line)
    # outer_spline_tag = gmsh.model.occ.add_spline(outer_tag_list)
    outer_loop_tag = gmsh.model.occ.add_curve_loop(line_list)
    cell_plane_tag = gmsh.model.occ.add_plane_surface([outer_loop_tag])
    gmsh.model.occ.synchronize()
    facets = gmsh.model.getBoundary([(2, cell_plane_tag)])
    facet_tag_list = []
    for i in range(len(facets)):
        facet_tag_list.append(facets[i][1])
    gmsh.model.add_physical_group(1, facet_tag_list, tag=facet_marker)
    gmsh.model.add_physical_group(2, [cell_plane_tag], tag=cell_marker)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        return curvRes
    
    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.generate(2)
    rank = MPI.COMM_WORLD.rank
    if use_tmp:
        tmp_folder = pathlib.Path(f"/root/tmp/tmp_2DCell_{rank}")
    else:
        tmp_folder = pathlib.Path(f"tmp_2DCell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "2DCell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()
    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf_facet, mf_cell = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 2, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    
    mesh = d.MeshView.create(mf_cell, cell_marker)
    bmesh = d.MeshView.create(mf_facet, facet_marker)

    n = d.FacetNormal(mesh)
    # estimate facet normals in CG2 for better smoothness and accuracy
    V = d.VectorFunctionSpace(mesh, "CG", 2)
    u = d.TrialFunction(V)
    v = d.TestFunction(V)
    ds = d.Measure("ds", mesh)
    a = d.inner(u, v) * ds
    lform = d.inner(n, v) * ds
    A = d.assemble(a, keep_diagonal=True)
    L = d.assemble(lform)

    A.ident_zeros()
    nh = d.Function(V)
    d.solve(A, nh.vector(), L)  # project facet normals onto CG1

    Vb = d.FunctionSpace(bmesh, "CG", 1)
    Vb_vec = d.VectorFunctionSpace(bmesh, "CG", 1)
    nb = d.interpolate(nh, Vb_vec)
    p, q = d.TrialFunction(Vb), d.TestFunction(Vb)
    dx = d.Measure("dx", bmesh)
    a = d.inner(p, q) * dx
    lform = d.inner(d.div(nb), q) * dx
    A = d.assemble(a, keep_diagonal=True)
    L = d.assemble(lform)
    A.ident_zeros()
    kappab = d.Function(Vb)
    d.solve(A, kappab.vector(), L)

    if incl_parallel:  # then include out of plane (parallel) curvature as well
        kappab_vec = kappab.vector().get_local()
        nb_vec = nb.vector().get_local()
        nb_vec = nb_vec.reshape((int(len(nb_vec) / 3), 3))
        normal_angle = np.arctan2(nb_vec[:, 0], nb_vec[:, 2])
        x = Vb.tabulate_dof_coordinates()
        parallel_curv = kappab_vec
        logic = x[:,0] != 0
        parallel_curv[logic] = np.sin(normal_angle[logic]) / x[logic, 0]
        # parallel_curv = np.sin(normal_angle) / x[:, 0]
        # inf_logic = np.isinf(parallel_curv)
        # parallel_curv[inf_logic] = kappab_vec[inf_logic]
        kappab.vector().set_local((kappab_vec + parallel_curv) / 2.0)
        kappab.vector().apply("insert")

    # return kappab at r,z values
    kappab.set_allow_extrapolation(True)
    # curv_vals = np.zeros(len(rVals))
    # for i in range(len(rVals)):
    #     curv_vals[i] = kappab(rVals[i], 0, zVals[i])
    return kappab

def compute_curvature_ellipse(
    rVals, zVals,
    incl_parallel: bool = False, 
):
    """
    Give analytical value for curvature on ellipse surface.
    If incl_parallel, parallel curvature is also included for axisymmetric case,
    assuming that r = 0 is the axis of symmetry (3D shape is a spheroid in this case,
    i.e. an ellipsoid with 2 axes sharing the same length)
    """
    aVal = (max(rVals) - min(rVals))/2
    bVal = (max(zVals) - min(zVals))/2
    r0 = (max(rVals) + min(rVals))/2
    z0 = (max(zVals) + min(zVals))/2
    rVals = rVals - r0
    zVals = zVals - z0

    curv_vals = np.zeros(len(rVals))
    for i in range(len(rVals)):
        tCur = np.arctan2(zVals[i]/bVal, rVals[i]/aVal)
        curv_vals[i] = aVal*bVal/(bVal**2 * np.cos(tCur)**2 + aVal**2 * np.sin(tCur)**2)**(3/2)
        if incl_parallel:
            phiCur = np.arctan((aVal/bVal)*np.tan(tCur))
            if rVals[i]!=0: # if =0, then the same meridional and parallel curv
                cp = np.sin(np.pi/2 - phiCur) / rVals[i]
                curv_vals[i] = (curv_vals[i] + cp)/2
    return curv_vals

def compute_curvature_ellipse_alt(
    rVals, zVals, aVal, bVal, r0, z0,
    incl_parallel: bool = False, 
):
    """
    Give analytical value for curvature on ellipse surface.
    If incl_parallel, parallel curvature is also included for axisymmetric case,
    assuming that r = 0 is the axis of symmetry (3D shape is a spheroid in this case,
    i.e. an ellipsoid with 2 axes sharing the same length)
    """
    rVals = rVals - r0
    zVals = zVals - z0

    curv_vals = np.zeros(len(rVals))
    curv_alt = np.zeros(len(rVals))
    for i in range(len(rVals)):
        tCur = np.arctan2(zVals[i]/bVal, rVals[i]/aVal)
        curv_vals[i] = aVal*bVal/(bVal**2 * np.cos(tCur)**2 + aVal**2 * np.sin(tCur)**2)**(3/2)
        if incl_parallel:
            phiCur = np.arctan((aVal/bVal)*np.tan(tCur))
            if rVals[i]!=0: # if =0, then the same meridional and parallel curv
                cp = np.sin(np.pi/2 - phiCur) / rVals[i]
                curv_vals[i] = (curv_vals[i] + cp)/2
            # alt, use explicit formula
            num = bVal*(4*aVal**2 + 2*(aVal**2 - bVal**2)*(rVals[i]/aVal)**2)
            den = np.sqrt(2)*aVal*(2*aVal**2 + 2*(aVal**2 - bVal**2)*(rVals[i]/aVal)**2)**(3/2)
            curv_alt[i] = num / den
    # assert np.all(curv_alt == curv_vals)
    return curv_vals

def calc_curv_NP(x,y,z,
                 a,b,z0Inner,xNP,yNP,nanopillars):
    radNP, hNP, pNP = nanopillars[:]
    xSteric = 0.05
    radNP += 2*xSteric # to avoid collision with PM
    hNP += 0.2
    zNP = hNP * np.ones_like(xNP)
    dist_vals = np.sqrt(np.power(xNP-x, 2) + np.power(yNP-y, 2)) 
    alpha = np.sqrt(1 - (x**2+y**2)/a**2)
    zCur = z0Inner - b*alpha
    hex = b*x/(a**2 * alpha)
    hey = b*y/(a**2 * alpha)
    hexy = b*x*y / (a**4 * alpha**3)
    hexx = b*x**2 / (a**4 * alpha**3) + b / (a**2 * alpha)
    heyy = b*y**2 / (a**4 * alpha**3) + b / (a**2 * alpha)
    hx = hex
    hy = hey
    hxy = hexy
    hxx = hexx
    hyy = heyy
    for i in range(len(xNP)):
        uRef = zNP[i] - zCur
        if z < zNP[i]+1e-3:
            dxCur = x - xNP[i]
            dyCur = y - yNP[i]
            rlocal = np.sqrt(dxCur**2 + dyCur**2)
            drCur = rlocal - radNP
            sigma1 = np.exp(-(drCur/0.2)**2)
            if dist_vals[i] < radNP+1e-3:
                curv_val = 0
                return curv_val
            else:
                hx += -hex*sigma1 - 2*uRef*drCur*dxCur*sigma1/(0.2**2 * rlocal)
                hy += -hey*sigma1 - 2*uRef*drCur*dyCur*sigma1/(0.2**2 * rlocal)
                xyCur = (-hexy*sigma1 + 2*hex*drCur*dyCur*sigma1/(0.2**2 * rlocal) + 2*hey*drCur*dxCur*sigma1/(0.2**2 * rlocal) +
                        (2*sigma1*uRef*drCur*dxCur*dyCur/(0.2**2 * rlocal**2))*(2*drCur/0.2**2 + 1/rlocal) - 
                        2*uRef*dxCur*dyCur*sigma1/(0.2**2 * rlocal**2))
                xyCurAlt = (-hexy*sigma1 + 2*hey*drCur*dxCur*sigma1/(0.2**2 * rlocal) + 2*hex*drCur*dyCur*sigma1/(0.2**2 * rlocal) +
                        (2*sigma1*uRef*drCur*dxCur*dyCur/(0.2**2 * rlocal**2))*(2*drCur/0.2**2 + 1/rlocal) - 
                        2*uRef*dxCur*dyCur*sigma1/(0.2**2 * rlocal**2))
                assert np.isclose(xyCur, xyCurAlt), f"{xyCur} vs {xyCurAlt}"
                hxy += xyCurAlt
                hxx += (-hexx*sigma1 + 4*hex*drCur*dxCur*sigma1/(0.2**2 * rlocal) + 
                        (2*sigma1*uRef*drCur*dxCur**2/(0.2**2 * rlocal**2))*(2*drCur/0.2**2 + 1/rlocal) - 
                        2*uRef*dxCur**2*sigma1/(0.2**2 * rlocal**2) - 2*uRef*drCur*sigma1/(0.2**2 * rlocal))
                hyy += (-heyy*sigma1 + 4*hey*drCur*dyCur*sigma1/(0.2**2 * rlocal) + 
                        (2*sigma1*uRef*drCur*dyCur**2/(0.2**2 * rlocal**2))*(2*drCur/0.2**2 + 1/rlocal) - 
                        2*uRef*dyCur**2*sigma1/(0.2**2 * rlocal**2) - 2*uRef*drCur*sigma1/(0.2**2 * rlocal))
    num = (1+hy**2)*hxx - 2*hx*hy*hxy + (1+hx**2)*hyy
    den = 2*(1 + hx**2 + hy**2)**(3/2)
    curv_val = num/den   
    return curv_val

def calc_stretch_NP(x,y,zDef,
                 a,b,z0Inner,xNP,yNP,nanopillars):
    radNP, hNP, pNP = nanopillars[:]
    xSteric = 0.05
    radNP += 2*xSteric # to avoid collision with PM
    hNP += 0.2
    zNP = hNP * np.ones_like(xNP)
    dist_vals = np.sqrt(np.power(xNP-x, 2) + np.power(yNP-y, 2)) 
    alpha = np.sqrt(1 - (x**2+y**2)/a**2)
    zCur = z0Inner - b*alpha
    z = zCur - z0Inner
    N = np.sqrt((x**2+y**2)/a**4 + z**2/b**4)
    uzx = 0
    uzy = 0
    uzz = 0
    for i in range(len(xNP)):
        uRef = zNP[i] - zCur
        if zDef < zNP[i]+1e-6:
            dxCur = x - xNP[i]
            dyCur = y - yNP[i]
            rlocal = np.sqrt(dxCur**2 + dyCur**2)
            drCur = rlocal - radNP
            sigma1 = np.exp(-(drCur/0.2)**2)
            if dist_vals[i] < radNP:
                stretch_val = np.abs(z)/(N*b**2)
                return stretch_val
            else:
                uzx += -2*(uRef*dxCur*drCur/(rlocal*0.2**2)) * sigma1
                uzy += -2*(uRef*dyCur*drCur/(rlocal*0.2**2)) * sigma1
                uzz += -sigma1
    stretch_val = ((1+uzz)/N)*np.sqrt((x/(a**2) + z*uzx/(b**2 * (1+uzz)))**2 +
                                      (y/(a**2) + z*uzy/(b**2 * (1+uzz)))**2 +
                                      (z/(b**2 * (1+uzz)))**2)

    return stretch_val


def calc_def_NP(x,y,z,a,b,z0Inner,xNP,yNP,nanopillars):
    radNP, hNP, pNP = nanopillars[:]
    xSteric = 0.05
    radNP += 2*xSteric # to avoid collision with PM
    hNP += 0.2
    zNP = hNP * np.ones_like(xNP)
    dist_vals = np.sqrt(np.power(xNP-x, 2) + np.power(yNP-y, 2)) 
    alpha = np.sqrt(1 - (x**2+y**2)/a**2)
    zCur = z0Inner - b*alpha
    uz = 0
    for i in range(len(xNP)):
        uRef = zNP[i] - zCur
        if z < zNP[i]+1e-6:
            dxCur = x - xNP[i]
            dyCur = y - yNP[i]
            rlocal = np.sqrt(dxCur**2 + dyCur**2)
            drCur = rlocal - radNP
            sigma1 = np.exp(-(drCur/0.2)**2)
            if dist_vals[i] < radNP:
                uz = uRef
                return uz
            else:
                uz += uRef*sigma1

    return uz
