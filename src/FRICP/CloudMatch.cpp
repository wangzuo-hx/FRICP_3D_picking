#include "CloudMatch.h"
#include "FRICP.h"
#include "ICP.h"

#include <omp.h>
#include <iostream>
#include <pcl/common/centroid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/region_growing.h>

#include <functional>

using namespace speedbot;

CloudMatch::CloudMatch() 
{
	pars_ptr = new ICP::Parameters();
	fricp_ptr = new FRICP<3>();

	thread_pool.reset(new ThreadPool(8));
	template_pc.reset(new pcl::PointCloud<pcl::PointXYZ>());

	template_pc_pose.setIdentity();
	_matched_trans.setIdentity();
	down_sample_size = 0.5f;
}

CloudMatch::~CloudMatch() 
{
	if (pars_ptr != NULL)
		delete pars_ptr;
	if (fricp_ptr != NULL)
		delete fricp_ptr;
}

Vertices CloudMatch::convertPointCloudToVertices(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	std::size_t point_n = cloud->size();

	Vertices vertices;
	vertices.resize(3, point_n);

	// for
#pragma omp parallel for
	for (int i = 0; i < cloud->size(); i++)
	{
		vertices(0, i) = cloud->points[i].x;
		vertices(1, i) = cloud->points[i].y;
		vertices(2, i) = cloud->points[i].z;
	}

	return vertices;
}

Vertices CloudMatch::convertPointNormalToVertices(pcl::PointCloud<pcl::Normal>::Ptr normal)
{
	std::size_t point_n = normal->size();

	Vertices vertices;
	vertices.resize(3, point_n);

	// for
#pragma omp parallel for
	for (int i = 0; i < normal->size(); i++)
	{
		vertices(0, i) = normal->points[i].normal_x;
		vertices(1, i) = normal->points[i].normal_y;
		vertices(2, i) = normal->points[i].normal_z;
	}

	return vertices;
}

Eigen::Matrix4f CloudMatch::matchPointClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud)
{
	/*pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(source_cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	pcl::PointCloud<pcl::Normal>::Ptr source_cloud_normals(new pcl::PointCloud<pcl::Normal>);
	ne.setKSearch(20);
	ne.compute(*source_cloud_normals);

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> target_ne;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr target_tree(new pcl::search::KdTree<pcl::PointXYZ>());
	target_ne.setInputCloud(target_cloud);
	target_ne.setSearchMethod(target_tree);
	pcl::PointCloud<pcl::Normal>::Ptr target_cloud_normals(new pcl::PointCloud<pcl::Normal>);
	target_ne.setKSearch(20);
	target_ne.compute(*target_cloud_normals);*/

	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_source_pc(new pcl::PointCloud<pcl::PointXYZ>());  //去除点云异常点
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;

	sor.setInputCloud(source_cloud);
	sor.setMeanK(20);
	sor.setStddevMulThresh(1.0);
	sor.filter(*filtered_source_pc);

	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target_pc(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> target_sor;

	target_sor.setInputCloud(target_cloud);
	target_sor.setMeanK(20);
	target_sor.setStddevMulThresh(1.0);
	target_sor.filter(*filtered_target_pc);


	Vertices source_vertices = convertPointCloudToVertices(filtered_source_pc);
	Vertices target_vertices = convertPointCloudToVertices(filtered_target_pc);

	//Vertices source_normals = convertPointNormalToVertices(source_cloud_normals);
	//Vertices target_normals = convertPointNormalToVertices(target_cloud_normals);

	// scaling     // 缩放和去均值处理，使源点云和目标点云具有相同的尺度和中心
	Eigen::Vector3d source_scale, target_scale;
	source_scale = source_vertices.rowwise().maxCoeff() - source_vertices.rowwise().minCoeff();
	target_scale = target_vertices.rowwise().maxCoeff() - target_vertices.rowwise().minCoeff();
	double scale = std::max(source_scale.norm(), target_vertices.norm());

	source_vertices /= scale;
	target_vertices /= scale;

	// De-mean
	VectorN source_mean, target_mean;
	source_mean = source_vertices.rowwise().sum() / double(source_vertices.cols());
	target_mean = target_vertices.rowwise().sum() / double(target_vertices.cols());
	source_vertices.colwise() -= source_mean;
	target_vertices.colwise() -= target_mean;

	// pars_ptr->nu_end_k = 1.0 / 6;
	// pars_ptr->max_icp = 10000;
	pars_ptr->f = ICP::WELSCH;
	pars_ptr->use_AA = true;
	// pars_ptr->stop = 1e-7;
	pars_ptr->print_output = false;
	pars_ptr->out_path = "./result.txt";
	// match point cloud
	fricp_ptr->point_to_point(source_vertices, target_vertices, source_mean, target_mean, *pars_ptr);


	MatrixXX res_trans = pars_ptr->res_trans;
	res_trans.block(0, 3, 3, 1) *= scale;

	Eigen::Matrix4f return_matrix = res_trans.cast<float>();

	std::cout << "return_matrix: " << std::endl;
	for (int i = 0; i < return_matrix.rows(); ++i) {
		for (int j = 0; j < return_matrix.cols(); ++j) {
			std::cout << return_matrix(i, j) << "\t";
		}
		std::cout << std::endl;
	}

	result_trans = res_trans;
	return return_matrix;

}

Eigen::Matrix4f CloudMatch::getLocalCoordinate(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	// Extract pca centroid
	Eigen::Vector4d pcaCentroid_shape_center_all;
	pcl::compute3DCentroid(*cloud, pcaCentroid_shape_center_all);

	// Extract covariance
	Eigen::Matrix3d covariance_shape_center_all;
	pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid_shape_center_all, covariance_shape_center_all);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver_shape_center_all(covariance_shape_center_all, Eigen::ComputeEigenvectors);
	Eigen::Matrix3d eigenVectorsPCA_shape_center_all = eigen_solver_shape_center_all.eigenvectors();
	Eigen::Vector3d eigenValuesPCA_shape_center_all = eigen_solver_shape_center_all.eigenvalues();

	std::cout << "eigen values pca : " << eigenValuesPCA_shape_center_all << std::endl;

	Eigen::Vector3d PCA_Biggest_all = eigenVectorsPCA_shape_center_all.col(2).normalized();
	Eigen::Vector3d PCA_Second_all = eigenVectorsPCA_shape_center_all.col(1).normalized();
	Eigen::Vector3d PCA_Smallest_all = eigenVectorsPCA_shape_center_all.col(0).normalized();

	MatrixXX trans(4,4);
	trans.setIdentity();

	trans.block(0, 0, 3, 1) = PCA_Biggest_all;
	trans.block(0, 1, 3, 1) = PCA_Second_all;
	trans.block(0, 2, 3, 1) = PCA_Smallest_all;
	trans.block(0, 3, 4, 1) = pcaCentroid_shape_center_all;

	Eigen::Matrix4f return_matrix = trans.cast<float>();

	return return_matrix;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CloudMatch::downSampleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float down_sample_size)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	 pcl::UniformSampling<pcl::PointXYZ> us;
	//pcl::VoxelGrid<pcl::PointXYZ> us;
	us.setInputCloud(cloud);
	 us.setRadiusSearch(down_sample_size);
	//us.setLeafSize(down_sample_size, down_sample_size, down_sample_size);
	us.filter(*filtered_cloud);

	return filtered_cloud;
}

float CloudMatch::getFinalConvergenceMse()
{
	return pars_ptr->convergence_gt_mse;
}

void CloudMatch::saveTemplatePC(std::string file_pth)
{
	auto fun = [file_pth](pcl::PointCloud<pcl::PointXYZ>::Ptr pc)
		{
			pcl::io::savePLYFile(file_pth, *pc);
		};

	thread_pool->enqueue<std::function<void(pcl::PointCloud<pcl::PointXYZ>::Ptr)>>(fun, this->template_pc);
}

bool CloudMatch::loadTemplatePC(std::string file_pth)
{
	pcl::io::loadPLYFile(file_pth, *(this->template_pc));

	bool res = false;
	if (!this->template_pc->empty())
		res = true;

	template_pc_pose = getLocalCoordinate(template_pc);

	return res;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CloudMatch::convertToPointCloud(const cv::Mat& rgb_img, const cv::Mat& depth_img, const cv::Rect& roi_rect)
{
	//get origin cloud on camera coordition
	cv::Mat depth_image;
	cv::Size sz = depth_img.size();
	depth_img.convertTo(depth_image, CV_32F, 1.0, 0.0);
	// m_3d_depth = depth_image.clone();
	int width = sz.width;
	int height = sz.height;
	const int image_size = width * height;
	std::vector<cv::Point2f> pts(image_size);

#pragma omp parallel for collapse(2)
	for (int r = 0; r < height; r++)
		for (int c = 0; c < width; c++)
		{
			pts[c + r * width].x = c;
			pts[c + r * width].y = r;
		}
	// distCoeff is [k1, k2, p1, p2, k3];
	std::vector<cv::Point2f> undistort_pts; // for undistorting RVC camera  depth
	cv::undistortPoints(pts, undistort_pts, this->cameraMatrix, this->distCoeff);      // 根据相机编号选择参数

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	float bad_point = std::numeric_limits<float>::quiet_NaN();

	int x = roi_rect.area() ? roi_rect.x : 0;
	int y = roi_rect.area() ? roi_rect.y : 0;
	int cols = roi_rect.area() ? roi_rect.width : width;
	int rows = roi_rect.area() ? roi_rect.height : height;
	cloud->points.resize(cols * rows);

#pragma omp parallel for collapse(2)
	for (int r = 0; r < rows; r++)
		for (int c = 0; c < cols; c++)
		{
			const float* depth_ptr = depth_image.ptr<float>(r + y);
			pcl::PointXYZ* pt = &(cloud->points[r * cols + c]);
			float z = (float)depth_ptr[c + x];
			if (std::isnan(z) || z > 1000.0)
			{
				pt->x = bad_point;
				pt->y = bad_point;
				pt->z = bad_point;
				continue;
			}
			pt->x = undistort_pts[c + x + (r + y) * width].x * z;
			pt->y = undistort_pts[c + x + (r + y) * width].y * z;
			pt->z = z;
		}

	cloud->height = 1;
	cloud->width = cloud->points.size();
	cloud->is_dense = false;

	std::vector<int> indices;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_without_nan(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::removeNaNFromPointCloud(*cloud, *cloud_without_nan, indices);
	// pcl::io::savePLYFile("./camera_coor_pointcloud.ply", *cloud_without_nan);

	return cloud_without_nan;
}

void CloudMatch::setCameraMatrix(cv::Mat& intrinsic)
{
	cameraMatrix = intrinsic.clone();
}

void CloudMatch::setDistCoeff(cv::Mat& distCoeff)
{
	this->distCoeff = distCoeff;
}

Eigen::Matrix4f CloudMatch::calCloudMatchingPose(cv::Mat& rgb, cv::Mat& depth, cv::Rect& cloud_roi)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr roi_pc = convertToPointCloud(rgb, depth, cloud_roi);

	pcl::PointCloud<pcl::PointXYZ>::Ptr downSampled_roi_pc = downSampleCloud(roi_pc, down_sample_size);
	pcl::PointCloud<pcl::PointXYZ>::Ptr downSampled_template_pc = downSampleCloud(template_pc, down_sample_size);

	_matched_trans = matchPointClouds(downSampled_template_pc, downSampled_roi_pc);

	Eigen::Matrix4f _result_matrix = _matched_trans * template_pc_pose;

	auto fun = [](pcl::PointCloud<pcl::PointXYZ> a, pcl::PointCloud<pcl::PointXYZ> b, Eigen::Matrix4f _trans) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_template_pc(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::transformPointCloud(a, *transformed_template_pc, _trans);
		pcl::io::savePLYFile("./transformed_template_pc.ply", *transformed_template_pc);
		pcl::io::savePLYFile("./downSampled_roi_pc.ply", b);
		};

	thread_pool->enqueue(fun, *downSampled_template_pc, *downSampled_roi_pc, _matched_trans);

	return _result_matrix;
}

void CloudMatch::extractTemplatePointCloud(cv::Mat& rgb, cv::Mat& depth, cv::Rect& roi_rect)
{
	auto pc = convertToPointCloud(rgb, depth, roi_rect);

	pc = downSampleCloud(pc, down_sample_size);

	// 对点云进行滤波处理(统计滤波)
	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_pc(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;

	sor.setInputCloud(pc);
	sor.setMeanK(50);
	sor.setStddevMulThresh(1.0);
	sor.filter(*filtered_pc);

	// 区域生长算法保留最大区域的点云数据
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>);
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
	normal_estimator.setSearchMethod(kdtree);
	normal_estimator.setInputCloud(filtered_pc);
	normal_estimator.setKSearch(50);
	normal_estimator.compute(*normal);

	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
	reg.setMinClusterSize(5000);
	reg.setMaxClusterSize(1000000);
	reg.setSearchMethod(kdtree);
	reg.setNumberOfNeighbours(30);
	reg.setInputCloud(filtered_pc);
	reg.setInputNormals(normal);
	reg.setSmoothnessThreshold(6.0 / 180.0 * M_PI);
	reg.setCurvatureThreshold(1.0);
	std::vector<pcl::PointIndices> clusters;
	reg.extract(clusters);

	std::sort(clusters.begin(), clusters.end(), [](pcl::PointIndices& a, pcl::PointIndices& b) {
		return a.indices.size() > b.indices.size();
		});

	pcl::copyPointCloud(*filtered_pc, clusters[0], *template_pc);
}

cv::Mat CloudMatch::drawCloudResultImg(const cv::Mat& rgb, Eigen::Matrix4f _trans)
{
	cv::Mat clone_rgb_img = rgb.clone();
	if (clone_rgb_img.channels() == 1)
		cv::cvtColor(clone_rgb_img, clone_rgb_img, cv::COLOR_GRAY2BGR);

	auto draw_pc = downSampleCloud(template_pc, 2.0);
	pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_draw_pc(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::transformPointCloud(*draw_pc, *transformed_draw_pc, _trans);

	pcl::io::savePLYFile("./temp.ply", *transformed_draw_pc);

	std::vector<cv::Point3f> objectPoints;
	objectPoints.clear();

	for (auto& p : transformed_draw_pc->points)
	{
		objectPoints.push_back(cv::Point3f(p.x, p.y, p.z));
	}

	std::vector<cv::Point2f> imagePoints;
	imagePoints.clear();

	cv::Mat rVec = (cv::Mat_<float>(3, 1) << 0, 0, 0);

	cv::Mat tVec = (cv::Mat_<float>(3, 1) << 0, 0, 0);

	cv::projectPoints(objectPoints, rVec, tVec, this->cameraMatrix, this->distCoeff, imagePoints);

	cv::Scalar color = cv::Scalar(0, 0, 255);
	if (!_trans.isIdentity())
		color = cv::Scalar(0, 255, 0);

	for (auto& p : imagePoints)
	{
		cv::circle(clone_rgb_img, cv::Point(p.x, p.y), 1, color, -1);
	}

	return clone_rgb_img;
}
