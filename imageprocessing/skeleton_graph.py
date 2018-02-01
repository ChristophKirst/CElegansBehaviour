# -*- coding: utf-8 -*-
"""
Skeleton to Graph transforms

This module provides fast routines to convert 3d skeletons
to graphs, optimize and plot them.
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np
import networkx as nx
#from mayavi import mlab


def ensure_zero_border(skeleton):
  """Ensure the skeleton is zero on the border pixels"""
  skeleton[0 ] = 0;
  skeleton[-1] = 0;
  skeleton[:, 0] = 0;
  skeleton[:,-1] = 0;
  if skeleton.ndim == 3:
    skeleton[:,:,  0] = 0;
    skeleton[:, :,-1] = 0;
  return skeleton;


def get_neighbourhood_3d(img,x,y,z):
  """Return the neighbourhoods of the indicated voxels
  
  Arguments:
    img (array): the 3d image
    x,y,z (n array): coordinates of the voxels to extract neighbourhoods from
  
  Returns:
    array (nx27 array): neighbourhoods
    
  Note:
    Assumes borders of the image are zero so that 0<x,y,z<w,h,d !
  """
  nhood = np.zeros((x.shape[0],27), dtype = bool);
  
  # calculate indices (if many voxels this is only 27 loops!)
  for xx in range(3):
    for yy in range(3):
      for zz in range(3):
        #w = _xyz_to_neighbourhood[xx,yy,zz];
        w = 9 * xx + 3 * yy + zz;
        idx = x+xx-1; idy = y+yy-1; idz = z+zz-1;
        nhood[:,w]=img[idx, idy, idz];
  
  nhood.shape = (nhood.shape[0], 3, 3, 3);
  nhood[:, 1, 1, 1] = 0;
  return nhood;
  

def get_neighbourhood_2d(img,x,y):
  """Return the neighbourhoods of the indicated voxels
  
  Arguments:
    img (array): the 2d image
    x,y (n array): coordinates of the voxels to extract neighbourhoods from
  
  Returns:
    array (nx9 array): neighbourhoods
    
  Note:
    Assumes borders of the image are zero so that 0<x,y<w,h !
  """
  nhood = np.zeros((x.shape[0],9), dtype = bool);
  
  # calculate indices (if many voxels this is only 9 loops!)
  for xx in range(3):
    for yy in range(3):
        #w = _xyz_to_neighbourhood[xx,yy,zz];
        w = 3 * xx + yy;
        idx = x+xx-1; idy = y+yy-1;
        nhood[:,w]=img[idx, idy];
  
  nhood.shape = (nhood.shape[0], 3, 3);
  nhood[:, 1, 1] = 0;
  return nhood;


#_nh2id_2d = np.array([[2 ** 0, 2 ** 1, 2 ** 2], [2 ** 3, 0, 2 ** 4], [2 ** 5, 2 ** 6, 2 ** 7]]).flatten();
#
#_nh2id_3d = np.array([[[2**25, 2**24, 2**23], [2**22, 2**21, 2**20], [2**19, 2**18, 2**17]],
#                      [[2**16, 2**15, 2**14], [2**13, 0    , 2**12], [2**11, 2**10, 2**9]],
#                      [[2**8 , 2**7 , 2**6 ], [2**5 , 2**4 , 2**3 ], [2**2 , 2**1 , 2**0]]]).flatten();


def skeleton_to_list(skeleton, with_neighborhoods = True):
  """Converts 3d skeleton to a list of coordinates"""
  dim = skeleton.ndim;
  if dim != 2 and dim != 3:
    raise ValueError('skeleton should be 2d or 3d, found %d' % dim);
  
  #get nonzero pixel + neighbourhoods + connectivity
  if dim == 2:
    x,y = np.where(skeleton);
    if with_neighborhoods:
      nh = get_neighbourhood_2d(skeleton, x,y);
    #nid  = nh * _nh2id_2d;  
    ids = np.transpose((x,y));  
  else:
    x,y,z = np.where(skeleton);
    if with_neighborhoods:
      nh = get_neighbourhood_3d(skeleton,x,y,z);
    #nid  = nh * _nh2id_3d;  
    ids = np.transpose((x,y,z));

  if with_neighborhoods:
      return (ids, nh);
  else:
      return ids;


from collections import OrderedDict

def skeleton_to_adjacency(skeleton):
  """Converts a binary skeleton image to a graph

  Arguments:
    skeleton (array): 2d/3d binary skeleton image
    
  Returns:
    dict: dict of adjacency information with entries node_id : [neighbours]
  """
  
  ids,nh = skeleton_to_list(skeleton, with_neighborhoods = True);
 
  adj = OrderedDict(); 
  if len(ids) == 0:
    return adj;
    #return nx.from_dict_of_lists(adj);
  elif len(ids) == 1:
    adj[tuple(ids[0])] = [];
    return adj;
    #return nx.from_dict_of_lists(adj);
  else:
    for i,pos in enumerate(ids):
      posnh = np.where(nh[i]);
      adj[tuple(pos)] = [tuple(p + pos -1) for p in np.transpose(posnh)]
    return adj;
    
    
def skeleton_to_nx_graph(skeleton):
  """Converts a binary skeleton image to a networkx graph
  
  Arguments:
    skeleton (array): 2d/3d binary skeleton image
    
  Returns:
    dict: dict of adjacency information with entries node_id : [neighbours]
  """
  
  ids,nh = skeleton_to_list(skeleton, with_neighborhoods = True);
  print('ids done...'); 
 
  if len(ids) == 0:
     return nx.Graph();
  elif len(ids) == 1:
    adj = {};
    adj[tuple(ids[0])] = [];
    return nx.from_dict_of_lists(adj);
  else:
    g = nx.Graph();
    for i,pos in enumerate(ids):
      if i % 500 == 0:
          print('%d/%d nodes constructed...' % (i, len(ids)));
      p = tuple(pos);
      g.add_node(p);
      posnh = np.where(nh[i]);
      for pp in np.transpose(posnh):
          g.add_edge(p, tuple(pp+pos-1));
    return g;


try:
  import graph_tool as gt;
except:
  pass


def skeleton_to_gt_graph(skeleton, with_coordinates = True, verbose = True):
  """Converts a binary skeleton image to a networkx graph
  
  Arguments:
    skeleton (array): 2d/3d binary skeleton image
    
  Returns:
    dict: dict of adjacency information with entries node_id : [neighbours]
  """
  dim = skeleton.ndim;
  shape =skeleton.shape;
  
  coords, nh = skeleton_to_list(skeleton, with_neighborhoods = True);
  nnodes = coords.shape[0];
  
  coordids = np.ravel_multi_index(coords.T, shape);  
  coordids2ids = { k:i for i,k in enumerate(coordids) };
    
  # create graph
  if verbose:
    print('creating graph...');
  g = gt.Graph(directed = False);
  g.add_vertex(nnodes);
  if with_coordinates:
      if verbose:
        print('creating coordinate properties...')
      vp = g.new_vertex_property('int', coords[:,0]);
      g.vertex_properties['x'] = vp;
      vp = g.new_vertex_property('int', coords[:,1]);
      g.vertex_properties['y'] = vp;
      if dim > 2:
          vp = g.new_vertex_property('int', coords[:,2]);
          g.vertex_properties['z'] = vp;
  
  for i, pos in enumerate(coords):
    if verbose and i % 1000 == 0:
      print('%d/%d nodes constructed...' % (i, len(coords)));
    #print 'nh'
    #print nh[i]
    posnh = np.transpose(np.where(nh[i]));
    #print 'pos'
    #print pos
    #print posnh
    if len(posnh) > 0:
      posnh = np.array(posnh + pos - 1);
      #print posnh
      #print posnh.shape
      ids = np.ravel_multi_index(posnh.T, shape);
      #print ids
      for j in [coordids2ids[k] for k in ids]:
        if i < j:
          g.add_edge(i,j);
  
  return g



#def skeleton_to_reduced_graph(skeleton):
#  """Reduced the skeketon to branch points and edges between them"""
#  dim = skeleton.ndim;
#  shape =skeleton.shape;
#  
#  coords, nh = skeleton_to_list(skeleton, with_neighborhoods = True);







def plot_nx_graph_3d(graph, radii = None,  colormap='jet', line_width = 2, opacity=.9):
  """Plot a 3d graph of the skeleton
  
  Arguments:
    radii: radii of the edges used in color code, if None uniform color
    
  Returns:
    mayavi scence
  """
  # get graph positions
  g2 = nx.convert_node_labels_to_integers(graph, label_attribute = 'xyz');
  xyz = np.array([x['xyz'] for x in g2.node.values()], dtype = 'int32');

  # scalar colors
  if radii is not None:
    scalars = np.array([radii[tuple(x)] for x in xyz], dtype = 'float32');
  else:
    #scalars = np.arange(5, xyz.shape[0]+5);
    scalars = np.ones(xyz.shape[0], dtype = 'float32');
  
  #pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
  #                    scalars,
  #                    scale_factor=node_size,
  #                    scale_mode='none',
  #                    colormap=graph_colormap,
  #                    resolution=20)

  pts = mlab.pipeline.scalar_scatter(xyz[:,0], xyz[:,1], xyz[:,2], scalars)
  
  pts.mlab_source.dataset.lines = np.array(g2.edges(), dtype = 'int32')
  pts.update()    
  
  #tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
  #lab.pipeline.surface(tube, color=edge_color)
  
  lines = mlab.pipeline.stripper(pts);
  mlab.pipeline.surface(lines, colormap = colormap, line_width = line_width, opacity = opacity)
  
  if radii is not None:
      mlab.colorbar(orientation = 'vertical', title='Radius [pix]');    
  
  mlab.axes()
  
  return lines


def plot_gt_graph_3d(graph, radii = None,  colormap='jet', line_width = 2, opacity=.9):
  """Plot a 3d graph of the skeleton
  
  Arguments:
    radii: radii of the edges used in color code, if None uniform color
    
  Returns:
    mayavi scence
  """
  # get graph positions
  x = np.array(graph.vertex_properties['x'].get_array(), dtype = 'int32');
  y = np.array(graph.vertex_properties['y'].get_array(), dtype = 'int32');
  z = np.array(graph.vertex_properties['z'].get_array(), dtype = 'int32');

  # scalar colors
  if radii is not None:
    #scalars = [radii[tuple(x)] for x in xyz];
    scalars = np.array(radii, dtype = 'float32');
  else:
    #scalars = np.arange(5, xyz.shape[0]+5);
    scalars = np.ones(x.shape[0], dtype = 'float32');
  
  #pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
  #                    scalars,
  #                    scale_factor=node_size,
  #                    scale_mode='none',
  #                    colormap=graph_colormap,
  #                    resolution=20)

  pts = mlab.pipeline.scalar_scatter(x, y, z, scalars)
  
  edgelist = np.vstack([np.array([e.source(), e.target()], dtype = 'int32') for e in graph.edges()]);
  pts.mlab_source.dataset.lines = edgelist;
  pts.update()    
  
  #tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
  #lab.pipeline.surface(tube, color=edge_color)
  
  lines = mlab.pipeline.stripper(pts);
  mlab.pipeline.surface(lines, colormap = colormap, line_width = line_width, opacity = opacity)
  
  if radii is not None:
      mlab.colorbar(orientation = 'vertical', title='Radius [pix]');    
  
  mlab.axes();
  
  return lines





def test2():
    import numpy as np
    import skeleton_graph as sg;
    from mayavi import mlab
    reload(sg);
    
    skel = np.load('test.npy');
    skel = sg.ensure_zero_border(skel);    
    
    mlab.figure()
    g = sg.skeleton_to_nx_graph(skel);
    sg.plot_nx_graph_3d(g);
    
    mlab.figure()
    g2 = sg.skeleton_to_gt_graph(skel);
    sg.plot_gt_graph_3d(g2);

    

def test():
  from importlib import reload
  import numpy as np
  from mayavi import mlab
  import skeleton_graph as sg
  reload(sg)
  
  skel = np.load('TestData/skeleton_big.npy');
  skel = skel[:150,:150, :150];
  skel = sg.ensure_zero_border(skel);
  print(skel.shape, skel.sum())
  
  mlab.figure()
  g = sg.skeleton_to_nx_graph(skel);
  sg.plot_nx_graph_3d(g);
  
  mlab.figure()
  g2 = sg.skeleton_to_gt_graph(skel);
  sg.plot_gt_graph_3d(g2);
