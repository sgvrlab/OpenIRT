/*

PLY ASCII -> Binary

Greg Turk, August 1994

---------------------------------------------------------------

Copyright (c) 1994 The Board of Trustees of The Leland Stanford
Junior University.  All rights reserved.   
  
Permission to use, copy, modify and distribute this software and its   
documentation for any purpose is hereby granted without fee, provided   
that the above copyright notice and this permission notice appear in   
all copies of this software and that you do not sell the software.   
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,   
EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY   
WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.   

*/

#include <stdio.h>
#include <math.h>
//#include <strings.h>
#include "ply.h"


/* user's vertex and face definitions for a polygonal object */

typedef struct Vertex {
  float x,y,z;
  float nx,ny,nz;
  void *other_props;       /* other properties */
} Vertex;

typedef struct Face {
  unsigned char nverts;    /* number of vertex indices in list */
  int *verts;              /* vertex index list */
  void *other_props;       /* other properties */
} Face;

char *elem_names[] = { /* list of the kinds of elements in the user's object */
  "vertex", "face"
};

PlyProperty vert_props[] = { /* list of property information for a vertex */
  {"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,x), 0, 0, 0, 0},
  {"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,y), 0, 0, 0, 0},
  {"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,z), 0, 0, 0, 0},
  {"nx", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,nx), 0, 0, 0, 0},
  {"ny", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,ny), 0, 0, 0, 0},
  {"nz", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,nz), 0, 0, 0, 0},
};

PlyProperty face_props[] = { /* list of property information for a vertex */
  {"vertex_indices", PLY_INT, PLY_INT, offsetof(Face,verts),
   1, PLY_UCHAR, PLY_UCHAR, offsetof(Face,nverts)},
};


/*** the PLY object ***/

static int nverts,nfaces;
static Vertex **vlist;
static Face **flist;
static PlyOtherElems *other_elements = NULL;
static PlyOtherProp *vert_other,*face_other;
static int nelems;
static char **elist;
static int num_comments;
static char **comments;
static int num_obj_info;
static char **obj_info;
static int file_type;

static int flip_order = 1;       /* flip order of vertices around the faces? */
static int flip_normals = 0;     /* flip vertex normals? */
static int has_nx,has_ny,has_nz; /* are normals in PLY file? */


/******************************************************************************
Main program.
******************************************************************************/

main(int argc, char *argv[])
{
  int i,j;
  char *s;
  char *progname;

  progname = argv[0];

/*
  while (--argc > 0 && (*++argv)[0]=='-') {
    for (s = argv[0]+1; *s; s++)
      switch (*s) {
        case 'n':
          flip_normals = 1;
          flip_order = 0;
          break;
        case 'b':
          flip_normals = 1;
          flip_order = 1;
          break;
        default:
          usage (progname);
          exit (-1);
          break;
      }
  }*/

  read_file();

  /* maybe flip the order of the vertices in each face */
  //if (flip_order)
  //  flip_vertex_order();

  /* maybe flip the vertex normals */
  //if (flip_normals)
  //  negate_normals();

  write_file();
}


/******************************************************************************
Print out usage information.
******************************************************************************/

usage(char *progname)
{
  fprintf (stderr, "usage: %s [flags] <in.ply >out.ply\n", progname);
  fprintf (stderr, "       -n (flip normals)\n");
  fprintf (stderr, "       -b (flip both normals and vertex order in faces)\n");
}


/******************************************************************************
Reverse the order of the vertices in each face.
******************************************************************************/

flip_vertex_order()
{
  int i,j;
  int temp;
  int num;
  Face *face;

  for (i = 0; i < nfaces; i++) {

    face = flist[i];
    num = face->nverts;

    /* swap early vertices with later ones */
    for (j = 0; j < num / 2; j++) {
      temp = face->verts[j];
      face->verts[j] = face->verts[num-j-1];
      face->verts[num-j-1] = temp;
    }
  }
}


/******************************************************************************
Negate the vertex normals.
******************************************************************************/

negate_normals()
{
  int i;

  for (i = 0; i < nverts; i++) {
    if (has_nx) vlist[i]->nx *= -1;
    if (has_ny) vlist[i]->ny *= -1;
    if (has_nz) vlist[i]->nz *= -1;
  }
}


/******************************************************************************
Read in the PLY file from standard in.
******************************************************************************/

read_file()
{
  int i,j,k;
  PlyFile *ply;
  int nprops;
  int num_elems;
  PlyProperty **plist;
  char *elem_name;
  float version;


  /*** Read in the original PLY object ***/


  ply  = ply_read (stdin, &nelems, &elist);
  ply_get_info (ply, &version, &file_type);

  for (i = 0; i < nelems; i++) {

    /* get the description of the first element */
    elem_name = elist[i];
    plist = ply_get_element_description (ply, elem_name, &num_elems, &nprops);

    if (equal_strings ("vertex", elem_name)) {

      /* see if vertex holds any normal information */
      has_nx = has_ny = has_nz = 0;
      for (j = 0; j < nprops; j++) {
        if (equal_strings ("nx", plist[j]->name)) has_nx = 1;
        if (equal_strings ("ny", plist[j]->name)) has_ny = 1;
        if (equal_strings ("nz", plist[j]->name)) has_nz = 1;
      }

      /* create a vertex list to hold all the vertices */
      vlist = (Vertex **) malloc (sizeof (Vertex *) * num_elems);
      nverts = num_elems;

      /* set up for getting vertex elements */

      ply_get_property (ply, elem_name, &vert_props[0]);
      ply_get_property (ply, elem_name, &vert_props[1]);
      ply_get_property (ply, elem_name, &vert_props[2]);
      if (has_nx) ply_get_property (ply, elem_name, &vert_props[3]);
      if (has_ny) ply_get_property (ply, elem_name, &vert_props[4]);
      if (has_nz) ply_get_property (ply, elem_name, &vert_props[5]);
      vert_other = ply_get_other_properties (ply, elem_name,
                     offsetof(Vertex,other_props));

      /* grab all the vertex elements */
      for (j = 0; j < num_elems; j++) {
        vlist[j] = (Vertex *) malloc (sizeof (Vertex));
        ply_get_element (ply, (void *) vlist[j]);
      }
    }
    else if (equal_strings ("face", elem_name)) {

      /* create a list to hold all the face elements */
      flist = (Face **) malloc (sizeof (Face *) * num_elems);
      nfaces = num_elems;

      /* set up for getting face elements */

      ply_get_property (ply, elem_name, &face_props[0]);
      face_other = ply_get_other_properties (ply, elem_name,
                     offsetof(Face,other_props));

      /* grab all the face elements */
      for (j = 0; j < num_elems; j++) {
        flist[j] = (Face *) malloc (sizeof (Face));
        ply_get_element (ply, (void *) flist[j]);
      }
    }
    else
      other_elements = ply_get_other_element (ply, elem_name, num_elems);
  }

  comments = ply_get_comments (ply, &num_comments);
  obj_info = ply_get_obj_info (ply, &num_obj_info);

  ply_close (ply);
}


/******************************************************************************
Write out the PLY file to standard out.
******************************************************************************/

write_file()
{
  int i,j,k;
  PlyFile *ply;
  int num_elems;
  char *elem_name;

  /*** Write out the final PLY object ***/


  ply = ply_write (stdout, 2, elem_names, PLY_BINARY_NATIVE);


  /* describe what properties go into the vertex and face elements */

  ply_element_count (ply, "vertex", nverts);
  ply_describe_property (ply, "vertex", &vert_props[0]);
  ply_describe_property (ply, "vertex", &vert_props[1]);
  ply_describe_property (ply, "vertex", &vert_props[2]);
  if (has_nx) ply_describe_property (ply, "vertex", &vert_props[3]);
  if (has_ny) ply_describe_property (ply, "vertex", &vert_props[4]);
  if (has_nz) ply_describe_property (ply, "vertex", &vert_props[5]);
  ply_describe_other_properties (ply, vert_other, offsetof(Vertex,other_props));

  ply_element_count (ply, "face", nfaces);
  ply_describe_property (ply, "face", &face_props[0]);
  ply_describe_other_properties (ply, face_other, offsetof(Face,other_props));

  ply_describe_other_elements (ply, other_elements);

  for (i = 0; i < num_comments; i++)
    ply_put_comment (ply, comments[i]);

  for (i = 0; i < num_obj_info; i++)
    ply_put_obj_info (ply, obj_info[i]);

  ply_header_complete (ply);

  /* set up and write the vertex elements */
  ply_put_element_setup (ply, "vertex");
  for (i = 0; i < nverts; i++)
    ply_put_element (ply, (void *) vlist[i]);

  /* set up and write the face elements */
  ply_put_element_setup (ply, "face");
  for (i = 0; i < nfaces; i++)
    ply_put_element (ply, (void *) flist[i]);

  ply_put_other_elements (ply);

  /* close the PLY file */
  ply_close (ply);
}

