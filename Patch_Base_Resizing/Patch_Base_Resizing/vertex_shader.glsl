#version 330

//attribute vec3 vertex_position;
//attribute vec3 vertex_color;
//attribute vec2 vertex_uv;
//varying vec3 fragment_color;
//varying vec2 fragment_vertex_uv;

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec2 vertex_uv;
uniform mat4 proj;
out vec2 fragment_vertex_uv;

void main () {
 // fragment_color = vertex_color;
  //fragment_vertex_uv = vec2(vertex_uv.x, vertex_uv.y);
	fragment_vertex_uv = vertex_uv;

	gl_Position =  proj * vec4(vertex_position, 1.0);
}