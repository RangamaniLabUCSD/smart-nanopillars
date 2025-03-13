# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import dolfin as d
from smart import mesh_tools
import pathlib
import numpy as np
import sys
sys.path.append("/root/shared/gitrepos/smart-nanopillars/utils")
import spread_cell_mesh_generation as mesh_gen
indentation = 2.8
nucScaleFactor = 0.8
nanopillars = [0.5, 3.0, 3.5]

# Test for equal volume and correct curvatures across meshes.

import os
mesh_folder = pathlib.Path("/root/shared/gitrepos/smart-comp-sci-data/meshes/nanopillars")
folder_list = os.listdir(mesh_folder)
nuc_vol = np.zeros(len(folder_list))
nm_sa = np.zeros(len(folder_list))
cyto_vol = np.zeros(len(folder_list))
pm_sa = np.zeros(len(folder_list))
for i in range(len(folder_list)):
    loaded = mesh_tools.load_mesh(mesh_folder / folder_list[i] / "spreadCell_mesh.h5")
    nuc_vol[i] = d.assemble(1.0*d.Measure("dx", d.create_meshview(loaded.mf_cell,2)))
    nm_sa[i] = d.assemble(1.0*d.Measure("dx", d.create_meshview(loaded.mf_facet,12)))
    cyto_vol[i] = d.assemble(1.0*d.Measure("dx", d.create_meshview(loaded.mf_cell,1)))
    pm_sa[i] = d.assemble(1.0*d.Measure("dx", d.create_meshview(loaded.mf_facet,10)))
    print(f"Current cyto vol is {cyto_vol[i]}")

# Test curvature computation. (used for plots shown in Fig 3)

# +
nanopillars = [0.5, 3.0, 3.5]
indentation = 1.8
loaded = mesh_tools.load_mesh(pathlib.Path(f"/root/shared/gitrepos/smart-comp-sci-data/meshes/nanopillars_finalCalc/nanopillars_indent{indentation}") / "spreadCell_mesh.h5")
facet_markers = loaded.mf_facet

zMax = max(loaded.mesh.coordinates()[:,2])
nanopillar_rad, nanopillar_height, nanopillar_spacing = nanopillars[:]
u_nuc, aInner, bInner = mesh_gen.get_u_nuc(nanopillars[1], max(loaded.mesh.coordinates()[:,2]), 
                                            0.8, indentation, nanopillars)
z0Inner = nanopillars[1] + bInner + 0.2 - indentation
xMax = np.ceil(aInner / nanopillars[2]) * nanopillars[2]
xNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
yNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
xNP, yNP = np.meshgrid(xNP, yNP)
xNP = xNP.flatten()
yNP = yNP.flatten()
mesh_cur = d.create_meshview(facet_markers, 12)
Vcur = d.FunctionSpace(mesh_cur, "P", 1)
curv_fcn = d.Function(Vcur)
uvec = curv_fcn.vector()[:]
NMCoords = Vcur.tabulate_dof_coordinates()[:]
for i in range(len(NMCoords)):
    uvec[i] = mesh_gen.calc_curv_NP(NMCoords[i][0],NMCoords[i][1],NMCoords[i][2],
                                                  aInner,bInner,z0Inner,xNP,yNP,nanopillars)
curv_fcn.vector().set_local(uvec)
curv_fcn.vector().apply("insert")
d.File(f"curv{indentation}.pvd") << curv_fcn
# -

# Test mesh generation.

radiusArray= [
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.5, 0.25, 0.5, 0.25,
 0.0, 0.0, 0.0, 0.0, 0.0]
pitchArray= [
 5.0, 2.5, 1.0, 5.0, 2.5, 1.0,
 5.0, 2.5, 5.0, 2.5,
 0.0, 0.0, 0.0, 0.0, 0.0]
heightArray=[
 1.0, 1.0, 1.0, 3.0, 3.0, 3.0,
 1.0, 1.0, 3.0, 3.0,
 0.0, 0.0, 0.0, 0.0, 0.0]
cellRadArray=[
 20.25, 18.52, 16.55, 19.93, 18.04, 15.39,
 20.01, 17.45, 18.06, 17.64,
 22.48, 18.08, 15.39, 14.18, 12.33]
EModArray=[
 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 
 10000000, 10000000, 10000000, 10000000,
 10000000, 14, 7, 3, 1]
indentationArray = [0, 0.4, 0.8, 1.0, 1.4, 1.8, 2.0, 2.4, 2.8]
innerParamsList = []
outerParamsList = []
for curIdx in [8]:#range(len(indentationArray)):
    hEdge = 0.5
    hInnerEdge = 0.5
    hNP = hEdge * 0.3
    indentation = indentationArray[curIdx]
    nanopillars = [0.5, 3.0, 3.5]
    # nanopillars = [radiusArray[curIdx], heightArray[curIdx], pitchArray[curIdx]]
    cell_mesh, facet_markers, cell_markers, substrate_markers, curv_markers, u_nuc, a_nuc = mesh_gen.create_3dcell(
                                                                        contactRad=15.5,#cellRadArray[curIdx],
                                                                        hEdge=hEdge, hInnerEdge=hInnerEdge, hNP=hNP,
                                                                        nanopillars=nanopillars,
                                                                        return_curvature=True,
                                                                        sym_fraction=1/8,
                                                                        nuc_compression=indentation)
    rValsOuter, zValsOuter, rValsInner, zValsInner, innerParams, u_nuc, rScale, outerParams = mesh_gen.get_shape_coords(
                                                15.5, nanopillars, indentation)
    NE_mesh = d.create_meshview(facet_markers, 12)
    V = d.FunctionSpace(NE_mesh, "P", 1)
    dfunc = d.Function(V)
    mesh_ref = cell_mesh
    bmesh = V.mesh()
    store_map = bmesh.topology().mapping()[mesh_ref.id()].vertex_map()
    values = dfunc.vector().get_local()
    for j in range(len(store_map)):
        cur_sub_idx = d.vertex_to_dof_map(V)[j]
        values[cur_sub_idx] = -curv_markers.array()[store_map[j]]
    dfunc.vector().set_local(values)
    dfunc.vector().apply("insert")
    d.File("curv2.8_retry.pvd") << dfunc
    innerParamsList.append(innerParams)
    outerParamsList.append(outerParams)
    print(innerParams)
    print(outerParams)
    print(f"Done with index {curIdx}")

# Print mesh statistics

# +
import dolfin as d
import sympy as sym
import numpy as np
import pathlib
from smart import config, mesh, model, mesh_tools
from smart.units import unit
from smart.model_assembly import (
    Compartment,
    Parameter,
    Reaction,
    Species,
    sbmodel_from_locals,
)
import sys
sys.path.insert(0, "/root/shared/gitrepos/smart-nanopillars/utils")
import spread_cell_mesh_generation as mesh_gen
mesh_folder = pathlib.Path("/root/shared/gitrepos/smart-comp-sci-data/meshes/nanopillars_movenuc/nanopillars_movenuc-1.4")

# Aliases - base units
uM = unit.uM
um = unit.um
molecule = unit.molecule
sec = unit.sec
dimensionless = unit.dimensionless
# Aliases - units used in model
D_unit = um**2 / sec
flux_unit = uM * um / sec
vol_unit = uM
surf_unit = molecule / um**2
Cyto = Compartment("Cyto", 3, um, 1)
PM = Compartment("PM", 2, um, 10)
Nuc = Compartment("Nuc", 3, um, 2)
NM = Compartment("NM", 2, um, 12)
PM.specify_nonadjacency(["NM", "Nuc"])
NM.specify_nonadjacency(["PM"])
parent_mesh = mesh.ParentMesh(
        mesh_filename=str(pathlib.Path(mesh_folder) / "spreadCell_mesh.h5"),
        mesh_filetype="hdf5",
        name="parent_mesh",
        curvature=pathlib.Path(mesh_folder) / "curvatures.xdmf",
        extra_keys=["subdomain0_2"]
    )
A = Species("A", 0.0, surf_unit, 0.0, D_unit, "PM")
B = Species("B", 0.0, vol_unit, 0.0, D_unit, "Cyto")
C = Species("C", 0.0, surf_unit, 0.0, D_unit, "NM")
D = Species("D", 0.0, vol_unit, 0.0, D_unit, "Nuc")
k1 = Parameter("k1", 1.0, 1 / sec)
k2 = Parameter("k2", 1.0, surf_unit / (vol_unit*sec))
a1 = Reaction("a1", ["A"], ["B"], param_map={"on":"k1","off":"k2"},
    explicit_restriction_to_domain="PM")
a2 = Reaction("a2", ["C"], ["D"], param_map={"on":"k1","off":"k2"},
    explicit_restriction_to_domain="NM")

pc, sc, cc, rc = sbmodel_from_locals(locals().values())
configCur = config.Config()
configCur.solver.update({"final_t": 1.0,"initial_dt": 0.1})

model_cur = model.Model(pc, sc, cc, rc, configCur, parent_mesh)
model_cur.initialize(initialize_solver=True)
model_cur.cc.print_to_latex()
# -

# Write substrate meshes.

import sys
import dolfin as d
sys.path.append("/root/shared/gitrepos/smart-nanopillars/utils")
import spread_cell_mesh_generation as mesh_gen
dmesh, mf2, mf3 = mesh_gen.create_substrate(nanopillars=[0.5, 1.0, 5.0], hEdge = 0.2, LBox=50, contact_rad=20.01)
# dmesh, mf2, mf3 = mesh_gen.create_substrate(nanopillars=[0.0, 0.0, 0.0], hEdge = 0.2, LBox=50)
d.ALE.move(dmesh, d.Expression(("0","0","-0.05"), degree=1))
d.File("substrate_500h1p5_omitNP.pvd") << dmesh

# Test curvature calculation and check against curvature expressions in sympy

# +
import sympy as sym
a = aInner
b = bInner

def calc_curv_NP(x,y,xNP,yNP,zNP,radNP,zTest):
    if len(zNP) == 1:
        zNP = zNP * np.ones_like(xNP)
    else:
        assert len(zNP) == len(xNP)
    dist_vals = np.sqrt(np.power(xNP-x, 2) + np.power(yNP-y, 2)) 
    uVal = 0
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
        if zTest < zNP[i]+1e-6:
            dxCur = x - xNP[i]
            dyCur = y - yNP[i]
            rlocal = np.sqrt(dxCur**2 + dyCur**2)
            drCur = rlocal - radNP
            sigma1 = np.exp(-(drCur/0.2)**2)
            if dist_vals[i] < radNP:
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

def calc_curv_NP_sym(x,y,xNP,yNP,zNP,radNP,zTest):
    if len(zNP) == 1:
        zNP = zNP * np.ones_like(xNP)
    else:
        assert len(zNP) == len(xNP)
    dist_vals = np.sqrt(np.power(xNP-x, 2) + np.power(yNP-y, 2)) 
    uVal = 0

    xSym, ySym = sym.symbols('x,y')
    he = z0Inner - b*sym.sqrt(1 - (xSym**2 + ySym**2)/a**2)
    hexSym = sym.diff(he, xSym)
    heySym = sym.diff(he, ySym)
    hexySym = sym.diff(hexSym, ySym)
    hexxSym = sym.diff(hexSym, xSym)
    heyySym = sym.diff(heySym, ySym)


    alpha = np.sqrt(1 - (x**2+y**2)/a**2)
    zCur = z0Inner - b*alpha
    subsDict = {xSym:x, ySym:y}
    hex = float(hexSym.subs(subsDict))
    hey = float(heySym.subs(subsDict))
    hexy = float(hexySym.subs(subsDict))
    hexx = float(hexxSym.subs(subsDict))
    heyy = float(heyySym.subs(subsDict))
    hx = hex
    hy = hey
    hxy = hexy
    hxx = hexx
    hyy = heyy
    for i in range(len(xNP)):
        uRef = zNP[i] - zCur
        if zTest < zNP[i]+1e-6:
            dxCur = x - xNP[i]
            dyCur = y - yNP[i]
            rlocal = np.sqrt(dxCur**2 + dyCur**2)
            drCur = rlocal - radNP
            sigma1 = np.exp(-(drCur/0.2)**2)
            if dist_vals[i] < radNP:
                curv_val = 0
                return curv_val
            else:
                rlocalSym = sym.sqrt((xSym-xNP[i])**2 + (ySym-yNP[i])**2)
                hSym = (zNP[i] - he) * sym.exp(-((rlocalSym - radNP)/0.2)**2)
                hxSym = sym.diff(hSym, xSym)
                hySym = sym.diff(hSym, ySym)
                hxySym = sym.diff(hxSym, ySym)
                hxxSym = sym.diff(hxSym, xSym)
                hyySym = sym.diff(hySym, ySym)
                hx += float(hxSym.subs(subsDict))
                hy += float(hySym.subs(subsDict))
                hxy += float(hxySym.subs(subsDict))
                hxx += float(hxxSym.subs(subsDict))
                hyy += float(hyySym.subs(subsDict))
                if np.abs(float(hxSym.subs(subsDict))) > 0.001:
                    print("pause") 
    num = (1+hy**2)*hxx - 2*hx*hy*hxy + (1+hx**2)*hyy
    den = 2*(1 + hx**2 + hy**2)**(3/2)
    curv_val = num/den   
    return curv_val

xVals = np.linspace(-a/5, a/5, 11)
yVals = np.linspace(-a/5, a/5, 11)
xGrid, yGrid = np.meshgrid(xVals, yVals)
curv = np.zeros_like(xGrid)
for i in range(len(xVals)):
    for j in range(len(yVals)):
        curv1 = calc_curv_NP_sym(xGrid[i][j],yGrid[i][j],xNP,yNP,
                           [nanopillar_height + 0.2],nanopillar_rad, 0)
        curv2 = calc_curv_NP(xGrid[i][j],yGrid[i][j],xNP,yNP,
                           [nanopillar_height + 0.2],nanopillar_rad, 0)
        assert np.isclose(curv1, curv2)
        curv[i][j] = curv1
        print(f"{i}, {j}")
from matplotlib import pyplot as plt
ax = plt.axes(projection ='3d')
surf = ax.plot_surface(xGrid, yGrid, curv,
                cmap = plt.get_cmap('viridis'),
                edgecolor = 'none')
ax.view_init(90, 0)
plt.colorbar(surf)
# -

# Compare curvature calculation over whole mesh in sympy vs. using analytical expressions.

# define nanopillar locations
xMax = np.ceil(a / nanopillar_spacing) * nanopillar_spacing
xNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
yNP = np.arange(-xMax, xMax+1e-12, nanopillar_spacing)
xNP, yNP = np.meshgrid(xNP, yNP)
xNP = xNP.flatten()
yNP = yNP.flatten()
mesh_cur = d.create_meshview(facet_markers, 12)
Vcur = d.FunctionSpace(mesh_cur, "P", 1)
curv_fcn = d.Function(Vcur)
uvec = curv_fcn.vector()[:]
curv_fcn_sym = d.Function(Vcur)
uvec_sym = curv_fcn_sym.vector()[:]
coords = Vcur.tabulate_dof_coordinates()
for i in range(len(coords)):
    uvec[i] = calc_curv_NP(coords[i,0],coords[i,1],xNP,yNP,
                           [nanopillar_height+0.2],nanopillar_rad, coords[i,2])
    uvec_sym[i] = calc_curv_NP_sym(coords[i,0],coords[i,1],xNP,yNP,
                           [nanopillar_height+0.2],nanopillar_rad, coords[i,2])
curv_fcn.vector().set_local(uvec)
curv_fcn.vector().apply("insert")
curv_fcn_sym.vector().set_local(uvec_sym)
curv_fcn_sym.vector().apply("insert")
d.File("curv.pvd") << curv_fcn
d.File("curv_sym.pvd") << curv_fcn_sym

# +
folder = pathlib.Path(f"/root/shared/gitrepos/smart-nanopillars/meshes/nanopillars_indent/nanopillars_indent2.8")
loaded = mesh_tools.load_mesh(folder / "spreadCell_mesh.h5")
dmesh = d.create_meshview(loaded.mf_facet, 12)
aInner = 6.805
bInner = 3.0
z0Inner = 3.4
xMax = np.ceil(aInner / nanopillars[2]) * nanopillars[2]
xNP = np.arange(-xMax, xMax+1e-12, nanopillars[2])
yNP = np.arange(-xMax, xMax+1e-12, nanopillars[2])
xNP, yNP = np.meshgrid(xNP, yNP)
xNP = xNP.flatten()
yNP = yNP.flatten()
V_NE = d.FunctionSpace(dmesh, "P", 1)
aNE = d.Function(V_NE)
avec = aNE.vector()[:]
coords = V_NE.tabulate_dof_coordinates()

for i in range(len(avec)):
    cur_coord = coords[i]
    avec[i] = mesh_gen.calc_stretch_NP(cur_coord[0],cur_coord[1],cur_coord[2],
                aInner,bInner,z0Inner,xNP,yNP,nanopillars)
    # a_cur = a_vector(V_scalar_coords[i])
    # avec[i] = np.sqrt(a_cur[0]**2 + a_cur[1]**2 + a_cur[2]**2)
aNE.vector().set_local(avec)
aNE.vector().apply("insert")
d.File('aTest2_d.pvd') << aNE
