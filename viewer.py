import numpy as np
import pyvista
import polars
import os
import pathlib
from PIL import Image
import tkinter as tk
from tkinter import filedialog as fd
from pathlib import Path
from tkinter import ttk


#grid_path = 'D:/DataWork/pypadResults/PAD/T1423070502.csv'
grid_path = 'D:/DataWork/pypadResults/PAD/T1423081501.csv'
#grid_path = 'D:/DataWork/SanCarlos/Pre/PAD/AZSCA_0018_2022_06301159.csv'
#clip_extents = 'auto'
clip_extents = [-11.3,-.5,11.3,.5]
#clip_extents = [-.5,-11.3,.5,11.3] # [minx, miny, minz, maxx, maxy, maxz], [minx, miny, maxx, maxy] or 'auto'
grid_path=Path(grid_path)
basename = grid_path.name
dem_path = Path('/'.join(grid_path.parts[:-2]) + '/DEM/' + basename)
points_path = Path('/'.join(grid_path.parts[:-2])+'/Points/'+basename)




def visualizeVoxels(grid_path, points_path, dem_path, clip_extents = 'auto', value_name = 'pad', max_value = 6,
                    color_map=['#386404','#386404','#386404','#386404','#386404', '#386404', '#4E3118']):

    global pl,pts_fill_actor,pts_fill_actor_leaf,pts_fill_actor_wood,pts_occluded_actor, filled_values

    grid = polars.read_csv(grid_path)
    cellSize = round(grid[1,0]-grid[0,0],5)

    if clip_extents == 'auto':
        extents = np.concatenate([grid[:,0:3].min().to_numpy().flatten(), grid[:,0:3].max().to_numpy().flatten()])
    else:
        if len(clip_extents)==4:
            extents = np.concatenate([grid[:, 0:3].min().to_numpy().flatten(), grid[:, 0:3].max().to_numpy().flatten()])
            extents[0:2] = clip_extents[0:2]
            extents[3:5] = clip_extents[2:4]
        elif len(clip_extents)==6:
            extents = np.array(clip_extents)
        else:
            raise ValueError("clip_extents needs to be 'auto' or a list of 4 or 6 floats")

    grid = grid.filter((grid[:, 0] >= extents[0]) & (grid[:, 0] <= extents[3]) &
                        (grid[:, 1] >= extents[1]) & (grid[:, 1] <= extents[4]) &
                        (grid[:, 2] >= extents[2]) & (grid[:, 2] <= extents[5]))

    points = polars.read_csv(points_path).to_numpy()
    points = points[(points[:, 0] >= extents[0]) & (points[:, 0] <= extents[3]) &
                    (points[:, 1] >= extents[1]) & (points[:, 1] <= extents[4])]

    pts_all = pyvista.PolyData(points[:, :3])
    demPoints = polars.read_csv(dem_path).to_numpy()
    demPoints = demPoints[(demPoints[:, 0] >= extents[0]) & (demPoints[:, 0] <= extents[3]) &
                          (demPoints[:, 1] >= extents[1]) & (demPoints[:, 1] <= extents[4])]

    shape = ((demPoints.max(0) - demPoints.min(0)) / cellSize).round().astype(int) + 1

    dem = pyvista.StructuredGrid(demPoints[:,0].reshape(shape[0:2]),
                                 demPoints[:,1].reshape(shape[0:2]),
                                 demPoints[:,2].reshape(shape[0:2])).texture_map_to_plane()



    filled_pts = pyvista.PolyData(grid.filter((polars.col('classification')>0) & (polars.col('hag')>=.1))[:,:3].to_numpy())
    filled_values = grid.filter((polars.col('classification')>0) & (polars.col('hag')>=.1))[value_name].to_numpy()
    filled_values = np.repeat(filled_values,6)
    occluded_pts = pyvista.PolyData(grid.filter(polars.col('classification')==-1)[:,:3].to_numpy())

    base_cube = pyvista.Cube((0,0,0),cellSize,cellSize,cellSize)
    scanner_leg = pyvista.Cylinder((0,0,-1),direction=(0,0,1),radius=.05,height=2)
    scanner_head = pyvista.Sphere(radius=.2)

    filled_glyphs = filled_pts.glyph(orient=False,scale=False,geom=base_cube)
    occluded_glyphs = occluded_pts.glyph(orient=False,scale=False,geom=base_cube)

    pl = pyvista.Plotter(lighting='three lights')
    #pl.enable_eye_dome_lighting()
    pl.enable_ssao(radius=1)
    pl.set_background('black', top='white')
    pl.view_isometric()
    pl.add_axes(interactive=True)
    pl.enable_terrain_style(mouse_wheel_zooms=True, shift_pans=True)
    pyvista.global_theme.full_screen = True

    if value_name == 'classification':
        filled_values = filled_values.astype(int)
        pts_fill_actor = pl.add_mesh(filled_glyphs, scalars=filled_values, opacity=1,
                                     show_scalar_bar=False, cmap=color_map, categories=True)
    else:
        pts_fill_actor = pl.add_mesh(filled_glyphs, clim=[0, max_value], scalars=filled_values, opacity=1,
                                     show_scalar_bar=False,cmap='YlGn')
    pts_occluded_actor = pl.add_mesh(occluded_glyphs,color=True,opacity=.25)
    pts_all_actor = pl.add_mesh(pts_all,scalars=points[:,2],clim=[-3, 12], show_scalar_bar=False)
    pts_all_actor.SetVisibility(False)

    pl.add_mesh(scanner_leg,color='black')
    pl.add_mesh(scanner_head,color='black')
    pl.add_mesh(dem,color='#8A5D36', smooth_shading=True)
    #pl.add_mesh(dem,texture=dirt_texture, smooth_shading=True)






    def set_filled_opacity(value):
        global pl
        global pts_fill_actor, pts_fill_actor_leaf, pts_fill_actor_wood,filled_values
        pl.remove_actor(pts_fill_actor)
        if value == 'classification':
            filled_values = filled_values.astype(int)
            pts_fill_actor = pl.add_mesh(filled_glyphs, scalars=filled_values, opacity=value, show_scalar_bar=True, cmap=color_map, categories=True)
        else:
            pts_fill_actor = pl.add_mesh(filled_glyphs, clim=[0, max_value], scalars=filled_values, opacity=value,show_scalar_bar=True,cmap='YlGn')
    pl.add_slider_widget(set_filled_opacity, [0, 1], title='Filled Voxel Opacity',pointa=(.1,.9),pointb=(.3,.9),value=1)

    def set_occluded_opacity(value):
        global pl
        global pts_occluded_actor
        pl.remove_actor(pts_occluded_actor)
        pts_occluded_actor = pl.add_mesh(occluded_glyphs, opacity=value)
    pl.add_slider_widget(set_occluded_opacity, [0, 1], title='Occluded Voxel Opacity',pointa=(.1,.7),pointb=(.3,.7),value=.1)

    def toggle_pts_vis(flag):
        pts_all_actor.SetVisibility(flag)
    pl.add_checkbox_button_widget(toggle_pts_vis, value=False, size=20)



    pl.show()
    #pl.export_gltf('vistest.gltf')
    #pl.export_html('vistest.html')

visualizeVoxels(grid_path,points_path,dem_path, clip_extents=clip_extents, value_name='pad')