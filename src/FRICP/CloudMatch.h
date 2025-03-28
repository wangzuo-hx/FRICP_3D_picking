#ifndef CLOUD_MATCH_H
#define CLOUD_MATCH_H

//#ifdef _WIN64
//#ifdef BUILD_DLL
//#define SBT_API __declspec(dllexport)
//#else
//#define SBT_API __declspec(dllimport)
//#endif  // MY_LIB_EXPORTS
//#elif defined(__linux__)
//#if __GNUC__ >= 4
//#define SBT_API   __attribute__((visibility("default")))
//#define SBT_LOCAL __attribute__((visibility("hidden")))
//#else
//#define SBT_API
//#define SBT_LOCAL
//#endif
//#endif  // _WIN32

#include <Eigen/Dense>
#include <pcl/common/common.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ThreadPool.h"

namespace ICP {
	class Parameters;
}

template<int N> class FRICP;

namespace speedbot
{
	using Scalar = double;
	using Vertices = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;
	using VectorN = Eigen::Matrix<Scalar, 3, 1>;

	template < int Rows, int Cols, int Options = (Eigen::ColMajor | Eigen::AutoAlign)>
	using MatrixT = Eigen::Matrix<Scalar, Rows, Cols, Options>;

	using MatrixXX = MatrixT<Eigen::Dynamic, Eigen::Dynamic>;

	class CloudMatch
	{
	public:
		CloudMatch();
		virtual ~CloudMatch();

		/**
		 * 将模板保存成点云文件.
		 * 
		 * \param file_pth
		 * \param cloud
		 */
		void saveTemplatePC(std::string file_pth);

		/**
		 * 载入模板点云用于计算匹配.
		 * 
		 * \param file_pth
		 * \param cloud
		 */
		bool loadTemplatePC(std::string file_pth);

		/**
		 * 计算出点云配准后的区域姿态.
		 * 
		 * \param rgb
		 * \param depth
		 * \param cloud_roi
		 * \return 
		 */
		Eigen::Matrix4f calCloudMatchingPose(cv::Mat& rgb, cv::Mat& depth, cv::Rect& cloud_roi);

		/**
		 * 将ROI框定区域的，自动处理部分点云数据，并设置为模板点云.
		 * 
		 * \param rgb
		 * \param depth
		 * \param roi_rect
		 */
		void extractTemplatePointCloud(cv::Mat& rgb, cv::Mat& depth, cv::Rect& roi_rect);
		
		/**
		 * 将模板点云利用project绘制到图片当中.
		 * 
		 * \param rgb
		 * \return 
		 */
		cv::Mat drawCloudResultImg(const cv::Mat& rgb, Eigen::Matrix4f _trans);

		/**
		 * 设置相机内参.
		 * 
		 * \param intrinsic
		 */
		void setCameraMatrix(cv::Mat& intrinsic);

		/**
		 * 设置畸变参数.
		 * 
		 * \param distCoeff
		 */
		void setDistCoeff(cv::Mat& distCoeff);

		/**
		 * 设置点云降采样的数值.
		 * 
		 * \param size
		 */
		void setDownSampleSize(float size)
		{
			down_sample_size = size;
		}

		// get final convergence gt mse
		float getFinalConvergenceMse();

		Eigen::Matrix4f getMatchingResult() { return _matched_trans; }

	// private:
		MatrixXX result_trans;

		ICP::Parameters* pars_ptr;
		FRICP<3>* fricp_ptr;

		// 线程池，可以用于点云保存
		std::shared_ptr<ThreadPool> thread_pool;

		// 模板的点云数据
		pcl::PointCloud<pcl::PointXYZ>::Ptr template_pc;
		Eigen::Matrix4f _matched_trans;
		Eigen::Matrix4f template_pc_pose;

		// 相机内参与畸变参数
		cv::Mat cameraMatrix;
		cv::Mat distCoeff;
		float down_sample_size;

		Vertices convertPointCloudToVertices(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

		Vertices convertPointNormalToVertices(pcl::PointCloud<pcl::Normal>::Ptr normal);

		// calculate transformation from source to target
		Eigen::Matrix4f matchPointClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
			pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud);
		/*void matchPointClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
			pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud, Eigen::Matrix4f& return_matrix);*/

		// get local coordinations
		Eigen::Matrix4f getLocalCoordinate(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

		// downsampled point clouds used for display
		pcl::PointCloud<pcl::PointXYZ>::Ptr downSampleCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float down_sample_size);

		/**
		 * 将深度图与RGB转换为点云.
		 * 
		 * \param rgb_img
		 * \param depth_img
		 * \return 
		 */
		pcl::PointCloud<pcl::PointXYZ>::Ptr convertToPointCloud(const cv::Mat& rgb_img, const cv::Mat& depth_img, const cv::Rect& roi_rect);


	};
}

#endif // !1

