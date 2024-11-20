#include "object.h"

struct Patch* patch;	// 要在後面函數給大小，所以用指標
struct Graph graph;
Mat Segment;    // 存各個patch的資訊
int patch_num;
double mesh_width;
double mesh_height;
int mesh_cols;
int mesh_rows;

// the target width and height to resize.
int target_width = 400;
int target_height = 200;

vector<Mesh> target_mesh;

double grid_size = 25.0;

Mat segmentation(Mat source)
{
    // variables of createGraphSegmentation() 
	double sigma = 1.5;			// sigma 對原圖像進行高斯濾波去噪
	float k = 200;				// k 控制合併後的區域的數量
	int min_size = 50;			// min : 後處理參數，分割後會有很多小區域，當區域像素點的個數小於min時，選擇與其差異最小的區域合併
	// Graph - Based Image Segmentation 分割器
	Ptr<ximgproc::segmentation::GraphSegmentation> segmentator = ximgproc::segmentation::createGraphSegmentation(sigma, k, min_size);
	segmentator->processImage(source, Segment);		// Segment an image and store output in dst(2nd variable).
													// Segments type = 4 = CV_32S, C1
	// find number of segments
	double min, max;
	minMaxLoc(Segment, &min, &max);
	int seg_num = (int)max + 1;
	cout << seg_num << " segments.\n";
	Mat result = Mat::zeros(Segment.rows, Segment.cols, CV_8UC3);

	// initial patch
	patch_num = seg_num;
	patch = (Patch*)malloc(sizeof(Patch) * patch_num);
	for (int i = 0; i < patch_num; ++i)
	{
		patch[i].id = i;
		patch[i].size = 0;
		patch[i].saliency = 0;
		patch[i].segment_color = Scalar(0, 0, 0);
		patch[i].significant_color = Scalar(0, 0, 0);
	}
	
	// set patch size
	for (int i = 0; i < Segment.rows; ++i)
	{
		for (int j = 0; j < Segment.cols; ++j)
		{
			patch[Segment.at<int>(i, j)].size += 1;
		}
	}
	
	// set patch color
	int tmp;
	for (int i = 0; i < source.rows; ++i)
	{
		for (int j = 0; j < source.cols; ++j)
		{
			tmp = Segment.at<int>(i, j);
			for (int index = 0; index < 3; ++index)
				patch[tmp].segment_color.val[index] += source.at<Vec3b>(i, j).val[index] / (double)patch[tmp].size;
		}
	}

	// set result mat
	for (int i = 0; i < Segment.rows; ++i)
	{
		for (int j = 0; j < Segment.cols; ++j)
		{
			for (int index = 0; index < 3; ++index)
				result.at<Vec3b>(i, j).val[index] = patch[Segment.at<int>(i, j)].segment_color.val[index];
		}
	}

	return result;
}

Mat set_significant(Mat source)
{
	Mat result = Mat::zeros(Segment.rows, Segment.cols, CV_8UC3);
	int tmp;
	// set significant color
	for (int i = 0; i < Segment.rows; ++i)
	{
		for (int j = 0; j < Segment.cols; ++j)
		{
			tmp = Segment.at<int>(i, j);
			for (int index = 0; index < 3; ++index)
				patch[tmp].significant_color.val[index] += source.at<Vec3b>(i, j).val[index] / (double)patch[tmp].size;
		}
	}

	// set saliency value
	for (int i = 0; i < patch_num; i++)
	{
		patch[i].saliency += (double)patch[i].significant_color.val[0];
		patch[i].saliency += (double)patch[i].significant_color.val[1] * 256;
		patch[i].saliency += (double)patch[i].significant_color.val[2] * 256 * 256;		// r 重要度較高
	}
	double min_saliency = 2e9;
	double max_saliency = -2e9;
	for (int i = 0; i < patch_num; i++)
	{
		min_saliency = min(min_saliency, patch[i].saliency);
		max_saliency = max(max_saliency, patch[i].saliency);
	}
	// normalized
	for (int i = 0; i < patch_num; ++i)
	{
		patch[i].saliency = (patch[i].saliency - min_saliency) / (max_saliency - min_saliency);
		//cout << patch[i].saliency << endl;
	}
	
	// set result
	for (int i = 0; i < Segment.rows; ++i)
	{
		for (int j = 0; j < Segment.cols; ++j)
		{	
			tmp = Segment.at<int>(i, j);
			for (int index = 0; index < 3; ++index)
				result.at<Vec3b>(i, j).val[index] = patch[tmp].significant_color.val[index];
		}
	}
	return result;
}

void setGraph()
{
	mesh_cols = ((Segment.cols - 1) / grid_size) + 1;
	mesh_rows = ((Segment.rows - 1) / grid_size) + 1;
	mesh_width = (double)(Segment.cols - 1) / (double)(mesh_cols - 1);
	mesh_height = (double)(Segment.rows - 1) / (double)(mesh_rows - 1);
	cout << "grid size : " << grid_size << endl;
	cout << "mesh cols : " << mesh_cols << endl;
	cout << "mesh rows : " << mesh_rows << endl;
	cout << "mesh width : " << mesh_width << endl;
	cout << "mesh height : " << mesh_height << endl;

	// set graph vertices
	for (int row = 0; row < mesh_rows; ++row)
		for (int col = 0; col < mesh_cols; ++col)
			graph.vertex.push_back(Point2d(col * mesh_width, row * mesh_height));
	
	// set graph edge
	int index;
	Mesh mesh;
	for (int row = 0; row < mesh_rows - 1; ++row)
	{
		for (int col = 0; col < mesh_cols - 1; ++col)
		{
			index = row * mesh_cols + col;
			int indices[4] = { index, index + mesh_cols, index + mesh_cols + 1, index + 1 };	// 逆時針(0:左上 1:左下 2:右下 3:右上)
			Edge edge;
			if (col != 0)	//不用把邊緣的edge算進
			{
				edge.pair_indice = make_pair(indices[0], indices[1]);
				graph.edge.push_back(edge);
			}
			edge.pair_indice = make_pair(indices[1], indices[2]);
			graph.edge.push_back(edge);
			edge.pair_indice = make_pair(indices[3], indices[2]);
			graph.edge.push_back(edge);
			if (row != 0)
			{
				edge.pair_indice = make_pair(indices[0], indices[3]);
				graph.edge.push_back(edge);
			}
			// mesh
			for (int i = 0; i < 4; i++)
			{
				int vertex_index = indices[i];
				mesh.node[i] = graph.vertex[vertex_index];
			}
			graph.mesh.push_back(mesh);
		}
	}
	/*for (int i = 0; i < graph.mesh.size(); i++)
		for (int j = 0; j < 4; j++)
			cout << graph.mesh[i].node[j] << endl;*/
}

void warping(int target_width, int target_height)
{
	if (target_height <= 0 || target_width <= 0)
	{
		cout << " target image size wrong !" << endl;
		exit(-1);
	}
	// set edge list of each patches
	vector<vector<int> > edge_index_list_of_patch(patch_num);
	for (int edge_index = 0; edge_index < graph.edge.size(); ++edge_index)
	{
		int v1_index = graph.edge[edge_index].pair_indice.first;
		int v2_index = graph.edge[edge_index].pair_indice.second;
		// 找到兩個vertex所屬的segments
		int patch_a = Segment.at<int>(graph.vertex[v1_index].y, graph.vertex[v1_index].x);
		int patch_b = Segment.at<int>(graph.vertex[v2_index].y, graph.vertex[v2_index].x);

		//cout << patch_a << " " << patch_b << endl;
		if (patch_a == patch_b)
		{
			edge_index_list_of_patch[patch_a].push_back(edge_index);
		}
		else
		{
			edge_index_list_of_patch[patch_a].push_back(edge_index);
			edge_index_list_of_patch[patch_b].push_back(edge_index);
		}
	}
	
	// set up cplex
	IloEnv env;
	IloNumVarArray x(env);
	IloExpr expr(env);
	// set variable x, y
	for (int i = 0; i < graph.vertex.size(); ++i)
	{
		x.add(IloNumVar(env, -IloInfinity, IloInfinity));	// x => 0 2 4 6 ...
		x.add(IloNumVar(env, -IloInfinity, IloInfinity));	// y => 1 3 5 7 ...
	}

	const double alpha = 0.8;
	const double width_ratio = (double)target_width / (Segment.cols - 1);
	const double height_ratio = (double)target_width / (Segment.rows - 1);
	const double ORIENTATION_WEIGHT = 12.0;

	// Patch transform constraint DTF
	for (int patch_index = 0; patch_index < edge_index_list_of_patch.size(); ++patch_index)
	{
		const vector<int>& edge_list = edge_index_list_of_patch[patch_index];
		if (!edge_list.size())
			continue;

		// set center edge => 隨意一條都行
		const Edge& center_edge = graph.edge[edge_list[0]];
		double c_x = graph.vertex[center_edge.pair_indice.first].x - graph.vertex[center_edge.pair_indice.second].x;
		double c_y = graph.vertex[center_edge.pair_indice.first].y - graph.vertex[center_edge.pair_indice.second].y;

		double matrix_a = c_x;
		double matrix_b = c_y;
		double matrix_c = c_y;
		double matrix_d = -c_x;

		double matrix_rank = matrix_a * matrix_d - matrix_b * matrix_c;
		if (fabs(matrix_rank) <= 1e-9)	//預防det(c) = 0
		{
			matrix_rank = (matrix_rank > 0 ? 1 : -1) * 1e-9;
		}

		double inverse_a = matrix_d / matrix_rank;
		double inverse_b = -matrix_b / matrix_rank;
		double inverse_c = -matrix_c / matrix_rank;
		double inverse_d = matrix_a / matrix_rank;

		double saliency = patch[patch_index].saliency;
		for (int i = 0; i < edge_list.size(); ++i)
		{
			const Edge& edge = graph.edge[edge_list[i]];

			double e_x = graph.vertex[edge.pair_indice.first].x - graph.vertex[edge.pair_indice.second].x;
			double e_y = graph.vertex[edge.pair_indice.first].y - graph.vertex[edge.pair_indice.second].y;

			double t_s = inverse_a * e_x + inverse_b * e_y;
			double t_r = inverse_c * e_x + inverse_d * e_y;

			// DST
			expr += alpha * saliency *
				(IloPower((x[edge.pair_indice.first * 2] - x[edge.pair_indice.second * 2]) -
					(t_s * (x[center_edge.pair_indice.first * 2] - x[center_edge.pair_indice.second * 2]) +
						t_r * (x[center_edge.pair_indice.first * 2 + 1] - x[center_edge.pair_indice.second * 2 + 1])), 2) +
					IloPower((x[edge.pair_indice.first * 2 + 1] - x[edge.pair_indice.second * 2 + 1]) - 
						(-t_r * (x[center_edge.pair_indice.first * 2] - x[center_edge.pair_indice.second * 2]) + 
							t_s * (x[center_edge.pair_indice.first * 2 + 1] - x[center_edge.pair_indice.second * 2 + 1])), 2));

			// DLT
			expr += (1 - alpha) * (1 - saliency) *
				(IloPower((x[edge.pair_indice.first * 2] - x[edge.pair_indice.second * 2]) -
					width_ratio * (t_s * (x[center_edge.pair_indice.first * 2] - x[center_edge.pair_indice.second * 2]) +
						t_r * (x[center_edge.pair_indice.first * 2 + 1] - x[center_edge.pair_indice.second * 2 + 1])), 2) +
					IloPower((x[edge.pair_indice.first * 2 + 1] - x[edge.pair_indice.second * 2 + 1]) -
						height_ratio * (-t_r * (x[center_edge.pair_indice.first * 2] - x[center_edge.pair_indice.second * 2]) +
							t_s * (x[center_edge.pair_indice.first * 2 + 1] - x[center_edge.pair_indice.second * 2 + 1])), 2));
		}
	}

	// Grid orientation constraint DOR
	for (int edge_index = 0; edge_index < graph.edge.size(); ++edge_index)
	{
		int v1 = graph.edge[edge_index].pair_indice.first;
		int v2 = graph.edge[edge_index].pair_indice.second;

		double delta_x = graph.vertex[v1].x - graph.vertex[v2].x;
		double delta_y = graph.vertex[v1].y - graph.vertex[v2].y;

		if (abs(delta_x) > abs(delta_y))	// Horizontal
			expr += ORIENTATION_WEIGHT * IloPower(x[v1 * 2 + 1] - x[v2 * 2 + 1], 2);
		else
			expr += ORIENTATION_WEIGHT * IloPower(x[v1 * 2] - x[v2 * 2], 2);
	}

	IloModel model(env);

	model.add(IloMinimize(env, expr));

	IloRangeArray hard_constraint(env);

	// Boundary constraint
	for (int row = 0; row < mesh_rows; ++row)
	{
		int index = row * mesh_cols;
		hard_constraint.add(x[index * 2] == graph.vertex[0].x);

		index = row * mesh_cols + mesh_cols - 1;
		hard_constraint.add(x[index * 2] == graph.vertex[0].x + target_width);
	}

	for (int col = 0; col < mesh_cols; ++col)
	{
		int index = col;
		hard_constraint.add(x[index * 2 + 1] == graph.vertex[0].y);

		index = (mesh_rows - 1) * mesh_cols + col;
		hard_constraint.add(x[index * 2 + 1] == graph.vertex[0].y + target_height);
	}

	// Avoid flipping
	for(int row = 0; row < mesh_rows; ++row)
		for (int col = 1; col < mesh_cols; ++col)
		{
			int right = row * mesh_cols + col;
			int left = row * mesh_cols + col - 1;
			hard_constraint.add((x[right * 2] - x[left * 2]) >= 1e-4);
		}

	for(int row = 1; row < mesh_rows; ++row)
		for (int col = 0; col < mesh_cols; ++col)
		{
			int down = row * mesh_cols + col;
			int up = (row - 1) * mesh_cols + col;
			hard_constraint.add((x[down * 2 + 1] - x[up * 2 + 1]) >= 1e-4);
		}

	model.add(hard_constraint);
	// solve
	IloCplex cplex(model);

	cplex.setOut(env.getNullStream());
	if (!cplex.solve()) {
		std::cout << "Failed to optimize the model.\n";
	}

	IloNumArray result(env);
	cplex.getValues(result, x);
	
	int index;
	Mesh mesh;
	for(int row = 0; row < mesh_rows - 1; ++row)
		for (int col = 0; col < mesh_cols - 1; ++col)
		{
			index = row * mesh_cols + col;
			int indices[4] = { index, index + mesh_cols, index + mesh_cols + 1, index + 1 };
			for (int i = 0; i < 4; i++)
			{
				Point2d node;
				node.x = result[indices[i] * 2];
				node.y = result[indices[i] * 2 + 1];
				mesh.node[i] = node;
			}
			target_mesh.push_back(mesh);
		}


	cout << "end cplex\n";
	model.end();
	cplex.end();
	env.end();
	
}


// opnegl callback
void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
	// sreen shot
	if (key == GLFW_KEY_S && action == GLFW_PRESS)
	{
		cv::Mat img(target_height, target_width, CV_8UC4);
		//use fast 4-byte alignment (default anyway) if possible
		glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
		//set length of one complete row in destination data (doesn't need to equal img.cols)
		glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());

		glReadPixels(0, 0, img.cols, img.rows, GL_BGRA, GL_UNSIGNED_BYTE, img.data);


		cv::flip(img, img, 0);
		cv::imwrite("./result/resizeImg.png", img);
	}
}


int main()
{
	
    Mat sourceImage, segmentImage, saliencyImage, significantImage;
	// read image
	sourceImage = imread("res/butterfly.jpg");
	segmentImage = imread("res/segmentation.jpg");
	saliencyImage = imread("res/saliency.jpg");
	if (!sourceImage.data || !segmentImage.data || !saliencyImage.data)
	{
		std::cout << "Fail to load sourceImage" << std::endl;
		return -1;
	}
	if (sourceImage.type() == CV_8UC3)
	{
		std::cout << "img type is rgb " << std::endl;
		std::cout << "source img width : " << sourceImage.size().width << std::endl;
		std::cout << "source img height : " << sourceImage.size().height << std::endl;
		
		cv::cvtColor(sourceImage, sourceImage, cv::COLOR_BGR2BGRA);
	}


    segmentImage = segmentation(segmentImage);
	significantImage = set_significant(saliencyImage);

	setGraph();

	warping(target_width, target_height);
	
	// create vertices to gl draw
	vector<GLVertex> glvertices;
	assert(graph.mesh.size() == target_mesh.size());
	for (int i = 0; i < target_mesh.size(); ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			GLVertex vt;
			vt.vertices = glm::vec3(target_mesh[i].node[j].x, target_mesh[i].node[j].y, 0);
			//vt.texcoords = glm::vec2(target_mesh[i].node[j].x / (float)sourceImage.cols, target_mesh[i].node[j].y / (float)sourceImage.rows);
			vt.texcoords = glm::vec2(graph.mesh[i].node[j].x / (float)sourceImage.size().width, 
				graph.mesh[i].node[j].y / (float)sourceImage.size().height);
			glvertices.push_back(vt);

			// change to two triangles
			if (j == 2)
			{
				// node 2
				vt.vertices = glm::vec3(target_mesh[i].node[j].x, target_mesh[i].node[j].y, 0);
				vt.texcoords = glm::vec2(graph.mesh[i].node[j].x / (float)sourceImage.size().width,
					graph.mesh[i].node[j].y / (float)sourceImage.size().height);
				glvertices.push_back(vt);
			}
			if (j == 3)
			{
				// node 0
				vt.vertices = glm::vec3(target_mesh[i].node[0].x, target_mesh[i].node[0].y, 0);
				vt.texcoords = glm::vec2(graph.mesh[i].node[0].x / (float)sourceImage.size().width,
					graph.mesh[i].node[0].y / (float)sourceImage.size().height);
				glvertices.push_back(vt);
			}
		}
	}
	

	// --------- openGL --------------
	glfwInit();
	// use openGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// create window
	GLFWwindow* window = glfwCreateWindow(target_width, target_height, "openGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Fail to create window" << std::endl;
		glfwTerminate();
		exit(0);
	}
	glfwMakeContextCurrent(window);

	//set key callback
	glfwSetKeyCallback(window, KeyCallback);

	if (glewInit() != GLEW_OK)
	{	//initial fail
		printf("init glew failed\n");
		glfwTerminate();
		return -1;
	}
	Shader* shader = new Shader("./vertex_shader.glsl", "./fragment_shader.glsl");
	unsigned int vao, vbo;
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);

	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, glvertices.size() * sizeof(glvertices[0]), glvertices.data(), GL_STATIC_DRAW);

	// position
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)0);
	// uv
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void*)offsetof(GLVertex, texcoords));

	glBindVertexArray(0);	// release vao

	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	
	if (sourceImage.elemSize() > 0)
	{
		cv::Mat glimg;
		cv::cvtColor(sourceImage, glimg, cv::COLOR_BGRA2RGBA);
		/*cv::imshow("1", sourceImage);
		cv::waitKey(0);*/

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sourceImage.cols, sourceImage.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, glimg.data);
	
		glGenerateMipmap(GL_TEXTURE_2D);
	}

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glm::mat4 proj = glm::mat4(1.0f);
	proj = glm::ortho(0.0f, (float)target_width, (float)target_height, 0.0f, -1.0f, 1.0f);
	while (!glfwWindowShouldClose(window))
	{
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		shader->use();

		shader->setMat4("proj", proj);
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glUniform1i(glGetUniformLocation(shader->ID, "texture"), 0);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, glvertices.size());

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);
	glfwTerminate();

}