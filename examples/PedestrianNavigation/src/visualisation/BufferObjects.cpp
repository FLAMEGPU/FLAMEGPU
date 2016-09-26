/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */
#include <stdlib.h>
#include <stdio.h>

#include "BufferObjects.h"
#include "CustomVisualisation.h"



void createVBO(GLuint* vbo, GLenum target, GLuint size)
{
    glGenBuffers( 1, vbo);
    glBindBuffer( target, *vbo);

    glBufferData( target, size, 0, GL_STATIC_DRAW);

    glBindBuffer( target, 0);

    checkGLError();
}

void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    *vbo = 0;
}

void createTBO(GLuint* tbo, GLuint* tex, GLuint size)
{

	glGenTextures(1, tex);
	glGenBuffers(1, tbo);

    glBindBuffer(GL_TEXTURE_BUFFER_EXT, *tbo);
    glBufferData(GL_TEXTURE_BUFFER_EXT, size, 0, GL_STATIC_DRAW);

	glBindTexture(GL_TEXTURE_BUFFER_EXT, *tex);
	glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, GL_RGBA32F_ARB, *tbo); 

	glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);
	glBindTexture(GL_TEXTURE_BUFFER_EXT, 0 );

    checkGLError();
}

void deleteTBO( GLuint* tbo)
{
    glBindBuffer( 1, *tbo);
    glDeleteBuffers( 1, tbo);

    *tbo = 0;
}