#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

#include <ilcplex/ilocplex.h>

#include <ilconcert/iloexpression.h>

#define GLEW_STATIC 
#include <GL/glew.h>	//glew要比flfw早引入
#include <GLFW/glfw3.h>
#include <bits/stdc++.h>
#include <tool/shader.h>

//矩陣運算
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
using namespace cv;

struct Edge
{
	pair<int, int> pair_indice;
};

struct Mesh
{
	Point2d node[5];		// 4點 左上 左下 右下 右上 
};

struct Graph
{
	 vector<Point2d> vertex;
	 vector<Edge> edge;
	 vector<Mesh> mesh;
};

// 以segment分成各個patchs， 賦予每個patch：id、size、segmentColor、saliencyValue、significantColor
struct Patch
{
	int id;
	int size;
	Scalar segment_color;
	Scalar significant_color;
	double saliency;
};

struct GLVertex
{
	glm::vec3 vertices;

	glm::vec2 texcoords;
};