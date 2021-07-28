#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc/types_c.h"

using namespace cv;
std::vector<cv::Mat> planes;
cv::Mat complexImage;

Matx33f rgb2yiq_mat(0.299f, 0.587f, 0.114f,
                    0.596f, -0.274f, -0.322f,
                    0.211f, -0.523f, 0.312f);

Matx33f yiq2rgb_mat(1.0f, 0.956f, 0.621f,
                    1.0f, -0.272f, -0.647f,
                    1.0f, -1.106f, 1.703f);

Mat rgb2yiq(const Mat &img)
{

    Mat img_out(img.size(), img.type());
    img_out = img.clone();
    for (int j = 0; j < img.rows; j++)
    {
        for (int i = 0; i < img.cols * 3; i += 3)
        {

            Vec3f pixel(img_out.at<float>(j, i + 2), img_out.at<float>(j, i + 1), img_out.at<float>(j, i));
            pixel = rgb2yiq_mat * pixel;


            for (int k = 0; k < 3; k++)
            {
                img_out.at<float>(j, i + 2 - k) = pixel[k];
            }
        }
    }
    return img_out;
}

Mat yiq2rgb(const Mat &img)
{

    Mat img_out(img.size(), img.type());
    img_out = img.clone();
    for (int j = 0; j < img.rows; j++)
    {
        for (int i = 0; i < img.cols * 3; i += 3)
        {

            Vec3f pixel(img_out.at<float>(j, i + 2), img_out.at<float>(j, i + 1), img_out.at<float>(j, i));
            pixel = yiq2rgb_mat * pixel;


            for (int k = 0; k < 3; k++)
            {
                if (pixel[k] > 255.0) pixel[k] = 255.0;
                else if (pixel[k] < 0.0) pixel[k] = 0.0;

                img_out.at<float>(j, i + 2 - k) = pixel[k];
            }
        }
    }
    return img_out;
}

string type2str(int type)
{
    string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch ( depth )
    {
    case CV_8U:
        r = "8U";
        break;

    case CV_8S:
        r = "8S";
        break;

    case CV_16U:
        r = "16U";
        break;

    case CV_16S:
        r = "16S";
        break;

    case CV_32S:
        r = "32S";
        break;

    case CV_32F:
        r = "32F";
        break;

    case CV_64F:
        r = "64F";
        break;

    default:
        r = "User";
        break;
    }
    r += "C";
    r += (chans + '0');
    return r;
}

void shiftDFT(cv::Mat image)
{
	image = image(Rect(0, 0, image.cols & -2, image.rows & -2));
	int cx = image.cols / 2;
	int cy = image.rows / 2;

	Mat q0 = Mat(image, Rect(0, 0, cx, cy));
	Mat q1 = Mat(image, Rect(cx, 0, cx, cy));
	Mat q2 = Mat(image, Rect(0, cy, cx, cy));
	Mat q3 = Mat(image, Rect(cx, cy, cx, cy));

	cv::Mat tmp = cv::Mat();
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

cv::Mat optimizeImageDim(cv::Mat image)
{
	// init
	cv::Mat padded;
	// get the optimal rows size for dft
	int addPixelRows = cv::getOptimalDFTSize(image.rows);
	// get the optimal cols size for dft
	int addPixelCols = cv::getOptimalDFTSize(image.cols);
	// apply the optimal cols and rows size to the image
	cv::copyMakeBorder(image, padded, 0, addPixelRows - image.rows, 0, addPixelCols - image.cols,
			cv::BORDER_CONSTANT, Scalar::all(0));

	return padded;
}


cv::Mat createOptimizedMagnitude(cv::Mat complexImage)
{
	// init
	std::vector<cv::Mat> newPlanes;
	cv::Mat mag = cv::Mat();
	// split the comples image in two planes
	cv::split(complexImage, newPlanes);
	// compute the magnitude
	cv::magnitude(newPlanes[0], newPlanes[1], mag);

	// move to a logarithmic scale
	cv::add(cv::Mat::ones(mag.size(), CV_32F), mag, mag);
	cv::log(mag, mag);
	// optionally reorder the 4 quadrants of the magnitude image
	shiftDFT(mag);
	// normalize the magnitude image for the visualization since both JavaFX
	// and OpenCV need images with value between 0 and 255
	// convert back to CV_8UC1
	mag.convertTo(mag, CV_8UC1);
	cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	return mag;
}

cv::Mat transformImage(cv::Mat image)
{
	// planes??????????????,???.
	if (!planes.empty()) {
		planes.clear();
	}
	// optimize the dimension of the loaded image
	cv::Mat padded = optimizeImageDim(image);
	padded.convertTo(padded, CV_32F);
	// prepare the image planes to obtain the complex image
	planes.push_back(padded);
	planes.push_back(cv::Mat::zeros(padded.size(), CV_32F));
	// prepare a complex image for performing the dft
	cv::merge(planes, complexImage);
	// dft
	printf("complexImage types %d\n", complexImage.type());
	cv::dft(complexImage, complexImage);

    // optimize the image resulting from the dft operation
    cv::Mat magnitude = createOptimizedMagnitude(complexImage);
    planes.clear();
    return magnitude;
}


void transformImageWithText(cv::Mat image)
{
    Scalar scalar(0, 0, 0, 0);
	if (!planes.empty()) {
		planes.clear();
	}
	// optimize the dimension of the loaded image
	cv::Mat padded = optimizeImageDim(image);
	padded.convertTo(padded, CV_32F);
	printf("padded types %d CV_32FC1 %d\n", padded.type(), CV_32F);
	// prepare the image planes to obtain the complex image
	planes.push_back(padded);
	planes.push_back(cv::Mat::zeros(padded.size(), CV_32F));
	// prepare a complex image for performing the dft
	cv::merge(planes, complexImage);
	printf("complexImage types %d\n", complexImage.type());
	// dft
	cv::dft(complexImage, complexImage);
	cv::Mat qr_code = cv::imread("qrcode.png", IMREAD_GRAYSCALE);
	cv::waitKey(0);

	for (int i=0;i<qr_code.rows;++i)
        for (int j=0;j<qr_code.cols;++j)
            if(qr_code.at<uchar>(i, j)==255)
            {
                complexImage.at<Vec2f>(10 + i, 10 + j)[0]=0;
                complexImage.at<Vec2f>(10 + i, 10 + j)[1]=0;
            }
	/*
	cv::putText(complexImage, "NUS", {40,60}, cv::FONT_HERSHEY_DUPLEX, 2.0, scalar,2);
	cv::putText(complexImage, "summer", {40,100}, cv::FONT_HERSHEY_DUPLEX, 2.0, scalar,2);
	cv::putText(complexImage, "work", {40,140}, cv::FONT_HERSHEY_DUPLEX, 2.0, scalar,2);
	cv::putText(complexImage, "shop", {40,180}, cv::FONT_HERSHEY_DUPLEX, 2.0, scalar,2);
	*/
	cv::flip(complexImage, complexImage, -1);
	for (int i=0;i<qr_code.rows;++i)
        for (int j=0;j<qr_code.cols;++j)
            if(qr_code.at<uchar>(i, j)==255)
            {
                complexImage.at<Vec2f>(10 + i, 10 + j)[0]=0;
                complexImage.at<Vec2f>(10 + i, 10 + j)[1]=0;
            }
	/*
	cv::putText(complexImage, "NUS", {40,60}, cv::FONT_HERSHEY_DUPLEX, 2.0, scalar,2);
	cv::putText(complexImage, "summer", {40,100}, cv::FONT_HERSHEY_DUPLEX, 2.0, scalar,2);
	cv::putText(complexImage, "work", {40,140}, cv::FONT_HERSHEY_DUPLEX, 2.0, scalar,2);
	cv::putText(complexImage, "shop", {40,180}, cv::FONT_HERSHEY_DUPLEX, 2.0, scalar,2);
	*/
	cv::flip(complexImage, complexImage, -1);
	planes.clear();
}

cv::Mat antitransformImage()
{
	cv::Mat invDFT = cv::Mat();
	cv::idft(complexImage, invDFT, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT, 0);
	cv::Mat restoredImage = cv::Mat();
	invDFT.convertTo(restoredImage, CV_8U);
	planes.clear();
	return restoredImage;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        return 0;
    }
    else
    {
        if (strcmp(argv[1], "enc") == 0)
        {
            //load image

            printf("read file %s\n", argv[2]);
            cv::Mat orig_img = cv::imread(argv[2]);
            //cv::Mat img = orig_img(Rect(orig_img.cols-512,orig_img.rows-512,512,512));
            cv::Mat img=orig_img;
            //imshow("haha",img);
            //cv::waitKey(0);
            vector<Mat>channels;
            split(img, channels);
            cv::Mat img1 = channels.at(0);
    		//cv::Mat img1 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
            transformImageWithText(img1);
            cv::Mat img2 = createOptimizedMagnitude(complexImage);

            cv::Mat img3 = antitransformImage();

            //cv::namedWindow("Matrix1", cv::WINDOW_AUTOSIZE);
            //cv::imshow("Matrix1", img1);

            //cv::namedWindow("Matrix2", cv::WINDOW_AUTOSIZE);
            //cv::imshow("Matrix2", img2);

            //cv::namedWindow("Matrix3", cv::WINDOW_AUTOSIZE);
            //cv::imshow("Matrix3", img3);

            channels.at(0) = img3;

            img1 = channels.at(1);
            transformImageWithText(img1);
            img2 = createOptimizedMagnitude(complexImage);
            img3 = antitransformImage();
            channels.at(1) = img3;

            img1 = channels.at(2);
            transformImageWithText(img1);
            img2 = createOptimizedMagnitude(complexImage);
            img3 = antitransformImage();
            channels.at(2) = img3;

            Mat img4;
            cv::merge(channels,img4);

            //img4.copyTo(orig_img(Rect(orig_img.cols-512,orig_img.rows-512,512,512)));


            cv::imwrite(argv[3], img4);
            //cv::imwrite("1_watermark.jpg", img2);
            //cv::imwrite("1_watermarked.jpg", img3);
            //cv::imwrite("watermarked.png", orig_img);

            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        if (strcmp(argv[1], "dec") == 0)
        {
            //load image
            //Point point(50, 100);
            //Scalar scalar(0, 0, 0, 0);
            printf("read file %s\n", argv[2]);
            cv::Mat img = cv::imread(argv[2]);

            waitKey(0);

            //cv::Mat img = orig_img;
            //cv::Mat img = orig_img(Rect(orig_img.cols-512,orig_img.rows-512,512,512));
            vector<Mat>channels;
            split(img, channels);
            printf("split done\n", argv[2]);
            cv::Mat img1 = channels.at(0);

    		//cv::Mat img1 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
            transformImage(img1);
            cv::Mat img2 = createOptimizedMagnitude(complexImage);

            //cv::Mat img3 = antitransformImage();

            channels.at(0) = img2;


            img1 = channels.at(1);
            transformImage(img1);
            img2 = createOptimizedMagnitude(complexImage);
            //img3 = antitransformImage();
            channels.at(1) = img2;

            img1 = channels.at(2);
            transformImage(img1);
            img2 = createOptimizedMagnitude(complexImage);
            //img3 = antitransformImage();
            channels.at(2) = img2;

            Mat img4;
            cv::merge(channels,img4);
            printf("merge done\n", argv[2]);
            /*
            cv::namedWindow("Matrix1", cv::WINDOW_AUTOSIZE);
            cv::imshow("Matrix1", img1);

            cv::namedWindow("Matrix2", cv::WINDOW_AUTOSIZE);
            cv::imshow("Matrix2", img2);
            */
            cv::imwrite(argv[3], img4);


            cv::waitKey(0);
            cv::destroyAllWindows();
        }

    }

    return 0;

}
