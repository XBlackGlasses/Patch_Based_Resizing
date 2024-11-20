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
#include <GL/glew.h>	//glew�n��flfw���ޤJ
#include <GLFW/glfw3.h>
#include <bits/stdc++.h>
#include <tool/shader.h>

//�x�}�B��
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
	Point2d node[5];		// 4�I ���W ���U �k�U �k�W 
};

struct Graph
{
	 vector<Point2d> vertex;
	 vector<Edge> edge;
	 vector<Mesh> mesh;
};

// �Hsegment�����U��patchs�A �ᤩ�C��patch�Gid�Bsize�BsegmentColor�BsaliencyValue�BsignificantColor
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