"""
Functions to create meshes for mechanotransduction example
"""

from typing import Tuple
import pathlib
import numpy as np
import dolfin as d
from mpi4py import MPI
from sympy.parsing.sympy_parser import parse_expr
from smart.mesh_tools import implicit_curve, compute_curvature, gmsh_to_dolfin, facet_topology

def create_3dcell(
    contactRad: float = 10.0,
    hEdge: float = 0,
    hInnerEdge: float = 0,
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
    roughness: Tuple[float, float] = [0, 0]
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
        nanopillars: tuple with nanopillar radius, height, spacing
        thetaExpr: String defining the theta dependence of the outer shape
        roughness: Tuple defining roughness parameters for pm and nm
    Returns:
        Tuple (mesh, facet_marker, cell_marker)
    Or, if return_curvature = True, Returns:
        Tuple (mesh, facet_marker, cell_marker, curvature_marker)
    """
    import gmsh

    # outerExpr25 = "(1 - z**4/(1500+z**4)) * (r**2 + z**2) + 0.4*(r**2 + (z+33.2)**2)*z**4 / (0.12 + z**4) - 625"
    # innerExpr25 = "(r/5.3)**2 + ((z-2.8)/2.4)**2 - 1"
    # outerExpr25 = "(1 - z**4/(10000+z**4)) * (r**2 + z**2) + 0.4*(r**2 + (z+15.2)**2)*z**4 / (3.0 + z**4) - 256"
    # outerExpr15 = "(1 - z**4/(1000+z**4)) * (r**2 + z**2) + 0.4*(r**2 + (z+15)**2)*z**4 / (10 + z**4) - 225"
    # innerExpr15 = "(r/5.3)**2 + ((z-7.8/2)/2.4)**2 - 1"
    # outerExpr13 = "(1 - z**4/(2000+z**4)) * (r**2 + z**2) + 0.4*(r**2 + (z+9.72)**2)*z**4 / (15 + z**4) - 169"
    # innerExpr13 = "(r/5.3)**2 + ((z-9.6/2)/2.4)**2 - 1"
    # outerExpr10 = "(1 - z**4/(35+z**4)) * (r**2 + z**2) + 0.94*(r**2 + (z+.01)**2)*z**4 / (30 + z**4) - 100"
    # innerExpr10 = "(r/5.3)**2 + ((z-5)/2.4)**2 - 1"
    # outerExprList = [outerExpr10, outerExpr13, outerExpr15, outerExpr25]
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
            xCurv*((nanopillar_rad+xSteric)**2 + (nanopillar_rad+xSteric)*xCurv*np.pi/2 + 
                   2*xCurv**2/3))
        zOffset = nanopillar_height
    else:
        nanopillar_vol = 0
        zOffset = 0
    
    targetVol = 480/scaleFactor**3 *4 + nuc_vol + nanopillar_vol
    rValsOuter, zValsOuter = shape_adj_axisymm(refParam, zOffset, targetVol)
    rValsOuter, zValsOuter = dilate_axisymm(rValsOuter, zValsOuter, targetVol)
    zMax = max(zValsOuter)

    rValsOuterClosed = np.concatenate((rValsOuter, -rValsOuter[::-1]))
    zValsOuterClosed = np.concatenate((zValsOuter,  zValsOuter[::-1]))
    curvFcnOuter = compute_curvature_1D(rValsOuterClosed, zValsOuterClosed, 
                                   curvRes=0.1, incl_parallel=True)

    innerExpr = get_inner(zOffset, zMax, nucScaleFactor)
    if not innerExpr == "":
        rValsInner, zValsInner = implicit_curve(innerExpr)
        aInner, bInner, r0Inner, z0Inner = get_inner_param(zOffset, zMax, nucScaleFactor)
        # rValsInnerClosed = np.concatenate((rValsInner, -rValsInner[-2:1:-1]))
        # zValsInnerClosed = np.concatenate((zValsInner,  zValsInner[-2:1:-1]))
        # curvValsInner = compute_curvature_ellipse(rValsInnerClosed, zValsInnerClosed, 
        #                                           incl_parallel=True)
        # curvValsInner = curvValsInner[0:len(rValsInner)]
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
        hInnerEdge = 0.2 * maxOuterDim if innerExpr == "" else 0.2 * maxInnerDim
    # Create the two axisymmetric body mesh using gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("3dcell")

    # first add outer body and revolve
    if thetaExpr != "":
        num_theta = 81
        if "rect" in thetaExpr:
            try:
                AR = float(thetaExpr[4:])
                a_rect = np.sqrt(np.pi * AR)
                b_rect = np.sqrt(np.pi / AR)
            except ValueError:
                raise ValueError("For rectangular pattern, " 
                    "thetaExpr must be rectAR, where AR is a number specifying " 
                    "the aspect ratio of the rectangle")
            theta_crit = np.arctan2(b_rect, a_rect)
            theta_incr = 2*np.pi / (num_theta-1)
            num_per_eighth = int(np.floor((num_theta-1) / 8))
            theta_range1 = np.linspace(0.0, theta_crit-theta_incr/4, num_per_eighth)
            theta_range2 = np.linspace(theta_crit+theta_incr/4, 
                                       np.pi-theta_crit-theta_incr/4, 2*num_per_eighth)
            theta_range3 = np.linspace(np.pi-theta_crit+theta_incr/4, 
                                       np.pi+theta_crit-theta_incr/4, 2*num_per_eighth)
            theta_range4 = np.linspace(np.pi+theta_crit+theta_incr/4, 
                                       2*np.pi-theta_crit-theta_incr/4, 2*num_per_eighth)
            theta_range5 = np.linspace(2*np.pi-theta_crit+theta_incr/4, 
                                       2*np.pi, num_per_eighth)
            thetaVec = np.concatenate([theta_range1, theta_range2, theta_range3, 
                                       theta_range4, theta_range5])
            thetaVec[-1] = 0.0 # for exactness, replace 2*pi with 0.0
            scaleVec = []
            for j in range(len(thetaVec)):
                if np.abs(np.tan(thetaVec[j])) <= b_rect / a_rect:
                    scaleVec.append(np.abs(a_rect / (2*np.cos(thetaVec[j]))))
                else:
                    scaleVec.append(np.abs(b_rect / (2*np.sin(thetaVec[j]))))
        else:
            thetaVec = np.linspace(0.0, 2*np.pi, num_theta)
            thetaVec[-1] = 0.0 # for exactness, replace 2*pi with 0.0
            thetaExprSym = parse_expr(thetaExpr)
            scaleVec = []
            for j in range(len(thetaVec)):
                scaleVec.append(float(thetaExprSym.subs({"theta": thetaVec[j]})))
        # define a smooth shape (ellipsoid) to average out sharp edges
        xVals = np.multiply(np.array(scaleVec), np.cos(np.array(thetaVec)))
        yVals = np.multiply(np.array(scaleVec), np.sin(np.array(thetaVec)))
        AR = (np.max(xVals) - np.min(xVals)) / (np.max(yVals) - np.min(yVals))
        a = np.sqrt(AR)
        b = 1/np.sqrt(AR)
        thetaExprSmooth = f"{a}*{b}/sqrt(({b}*cos(theta))**2 + ({a}*sin(theta))**2)"
        thetaExprSmooth = parse_expr(thetaExprSmooth)
        scaleVecSmooth = []
        for j in range(len(thetaVec)):
            scaleVecSmooth.append(float(thetaExprSmooth.subs({"theta": thetaVec[j]})))
    else:
        thetaVec = [0]
    cell_plane_tag = []
    bottom_point_list = []
    outer_spline_list = []
    edge_surf_list = []
    edge_segments = []
    top_point = gmsh.model.occ.add_point(0.0, 0.0, zValsOuter[0])
    all_points_list = [top_point]
    if thetaExpr != "":
        # define theta dependence
        rand_rough = np.random.rand(20,20)
        rand_rough_neg = np.random.rand(20,20)
        for j in range(len(thetaVec)):
            if j == (len(thetaVec)-1):
                outer_spline_list.append(outer_spline_list[0])
                bottom_point_list.append(bottom_point_list[0])
            else:
                outer_tag_list = []
                for i in range(len(rValsOuter)):
                    if i == 0:
                        outer_tag_list.append(top_point)
                    else:
                        # average out sharp edges (cell becomes more and more ellipsoidal
                        # away from the substrate)
                        xValRef = rValsOuter[i]*scaleVec[j]*np.cos(thetaVec[j])
                        yValRef = rValsOuter[i]*scaleVec[j]*np.sin(thetaVec[j])
                        xValSmooth = rValsOuter[i]*scaleVecSmooth[j]*np.cos(thetaVec[j])
                        yValSmooth =rValsOuter[i]*scaleVecSmooth[j]*np.sin(thetaVec[j])
                        zScale1 = (zMax - zValsOuter[i])/zMax
                        zScale2 = zValsOuter[i]/zMax
                        xCur, yCur, zCur = (zScale1*xValRef + zScale2*xValSmooth, 
                                            zScale1*yValRef + zScale2*yValSmooth, 
                                            zValsOuter[i])
                        if roughness[0] > 0:
                            # define deformation field over surface from smooth function
                            max_wavelength = 5.0
                            zDev = 0
                            for m in range(20):
                                for n in range(20):
                                    xArg = 2*np.pi*xCur*(m+1)/max_wavelength
                                    yArg = 2*np.pi*yCur*(n+1)/max_wavelength
                                    u1 = 1#rand_rough[m,n]
                                    u2 = np.sqrt(1 - u1**2)
                                    curMag = 2*roughness[0] / ((m+1)**2 + (n+1)**2) 
                                    zDev = zDev + curMag * (u1 * (np.cos(xArg)*np.cos(yArg) + np.sin(xArg)*np.sin(yArg)) +
                                                            u2 * (np.sin(xArg)*np.cos(yArg) + np.cos(xArg)*np.sin(yArg)))
                                    xArg_neg = 2*np.pi*xCur*(-m-1)/max_wavelength
                                    yArg_neg = 2*np.pi*yCur*(-n-1)/max_wavelength
                                    u1_neg = 1#rand_rough_neg[m,n]
                                    u2_neg = np.sqrt(1 - u1**2)
                                    zDev = zDev + curMag * (u1_neg * (np.cos(xArg_neg)*np.cos(yArg_neg) + np.sin(xArg_neg)*np.sin(yArg_neg)) +
                                                            u2_neg * (np.sin(xArg_neg)*np.cos(yArg_neg) + np.cos(xArg_neg)*np.sin(yArg_neg)))
                            # dev_vec = roughness[0] * np.random.randn(1)
                            # xCur = xCur + dev_vec[0]
                            # yCur = yCur + dev_vec[1]
                            if i == len(rValsOuter)-1:
                                zCur = 0
                            else:
                                zCur = max(zCur + zDev, 0) # z cannot be less than zero
                        
                        cur_tag = gmsh.model.occ.add_point(xCur, yCur, zCur)
                        outer_tag_list.append(cur_tag)
                        all_points_list.append(cur_tag)
                outer_spline_list.append(gmsh.model.occ.add_spline(outer_tag_list))
                bottom_point_list.append(outer_tag_list[-1])
            if j > 0:
                edge_tag = gmsh.model.occ.add_line(bottom_point_list[j-1], bottom_point_list[j])
                edge_loop_tag = gmsh.model.occ.add_curve_loop(
                    [outer_spline_list[j], outer_spline_list[j-1], edge_tag])
                edge_surf_list.append(gmsh.model.occ.add_bspline_filling(edge_loop_tag, type="Curved"))
                edge_segments.append(edge_tag)
        # now define total outer shape from edge_segments and edge_surf_list    
        bottom_loop = gmsh.model.occ.add_curve_loop(edge_segments)
        bottom_surf = gmsh.model.occ.add_plane_surface([bottom_loop])
        cur_surf_loop = gmsh.model.occ.add_surface_loop([bottom_surf, *edge_surf_list])
        outer_shape = gmsh.model.occ.add_volume([cur_surf_loop])
        outer_shape = [(3, outer_shape)]
    else:
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

    # need to fix labeling of facets!
    if innerExpr == "":
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
        inner_tag_list = []
        inner_line_list = []
        for i in range(len(rValsInner)):
            cur_tag = gmsh.model.occ.add_point(rValsInner[i], 0, zValsInner[i])
            inner_tag_list.append(cur_tag)
            if i > 0:
                cur_line = gmsh.model.occ.add_line(inner_tag_list[-2], inner_tag_list[-1])
                inner_line_list.append(cur_line)
        inner_spline_tag = gmsh.model.occ.add_spline(inner_tag_list)
        symm_inner_tag = gmsh.model.occ.add_line(inner_tag_list[0], inner_tag_list[-1])
        inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, symm_inner_tag])
        inner_plane_tag = gmsh.model.occ.add_plane_surface([inner_loop_tag])
        inner_shape = gmsh.model.occ.revolve([(2, inner_plane_tag)], 0, 0, 0, 0, 0, 1, 2*np.pi*sym_fraction)
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
                np.isclose(thetaCur, 0.0) or np.isclose(thetaCur, 2*np.pi*sym_fraction)):
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
                        curv_markers.set_value(v.index(), -(cm+cp)/2) 
                    elif zVal > xCurv: # cylinder
                        curv_markers.set_value(v.index(), -0.5/nanopillar_rad)
                    elif zVal > 0: # curved to substrate
                        cosCur = (xCurv - zVal)/xCurv
                        sinCur = np.sqrt(1 - cosCur**2)
                        cm = 1/xCurv
                        cp = sinCur / (nanopillar_rad + xSteric + xCurv*(1-sinCur))
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
                    rVal = np.sqrt(v.midpoint().x()**2 + v.midpoint().y()**2)
                    zVal = v.midpoint().z()
                    curv_val = compute_curvature_ellipse_alt(np.array([rVal]), np.array([zVal]), 
                                                             aInner, bInner, 
                                                             r0Inner, z0Inner, incl_parallel=True)
                    curv_markers.set_value(v.index(), curv_val[0])
        return (dmesh, mf2, mf3, substrate_markers, curv_markers)
    else:
        return (dmesh, mf2, mf3, substrate_markers)


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
    rValsOuter, zValsOuter = shape_adj_axisymm(refParam, zOffset, targetVol)
    rValsOuter, zValsOuter = dilate_axisymm(rValsOuter, zValsOuter, targetVol)
    zMax = max(zValsOuter)

    innerExpr = get_inner(zOffset, zMax, nucScaleFactor)

    if return_curvature:
        # create full mesh for curvature analysis and then map onto half mesh
        # if half_cell_with_curvature is True
        half_cell_with_curvature = half_cell
        half_cell = False

    if not innerExpr == "":
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
        hInnerEdge = 0.2 * maxOuterDim if innerExpr == "" else 0.2 * maxInnerDim
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

def dilate_axisymm(rVec, zVec, volTarget):
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
    while np.abs(volCur-volTarget) > 0.001*volTarget:
        for i in range(len(rVec)-1):
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
        
    return (rVec, zVec)

def get_inner(zOffset, zMax, scaleFactor):
    # zMid = (zOffset + zMax) / 2
    zRad = 2.4/scaleFactor
    rRad = 5.3/scaleFactor
    if (zMax - zOffset) <= (zRad*2 + 1.0):
        zRad = (zMax-zOffset)/2 - 1.0
        rRad = np.sqrt((5.3**2 * 2.4/scaleFactor**3) / zRad)
        if zRad < 0:
            raise ValueError("Nucleus does not fit")
        # return ""
    zMid = zOffset + zRad + 0.2
    return f"(r/{rRad})**2 + ((z-{zMid})/{zRad})**2 - 1"

def get_inner_param(zOffset, zMax, scaleFactor):
    zRad = 2.4/scaleFactor
    rRad = 5.3/scaleFactor
    if (zMax - zOffset) <= (zRad*2 + 1.0):
        zRad = (zMax-zOffset)/2 - 1.0
        rRad = np.sqrt((5.3**2 * 2.4/scaleFactor**3) / zRad)
        if zRad < 0:
            raise ValueError("Nucleus does not fit")
    zMid = zOffset + zRad + 0.2
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
    for i in range(len(rVals)):
        tCur = np.arctan2(zVals[i]/bVal, rVals[i]/aVal)
        curv_vals[i] = aVal*bVal/(bVal**2 * np.cos(tCur)**2 + aVal**2 * np.sin(tCur)**2)**(3/2)
        if incl_parallel:
            phiCur = np.arctan((aVal/bVal)*np.tan(tCur))
            if rVals[i]!=0: # if =0, then the same meridional and parallel curv
                cp = np.sin(np.pi/2 - phiCur) / rVals[i]
                curv_vals[i] = (curv_vals[i] + cp)/2
    return curv_vals
