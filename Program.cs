using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Security.Cryptography.X509Certificates;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Flann;
using System.Drawing;
using Emgu.CV.Util;
using System.Diagnostics;
using Emgu.CV.Structure;
using Emgu.CV.XFeatures2D;
using static System.Net.Mime.MediaTypeNames;
using static Emgu.CV.Ccm.ColorCorrectionModel;
using System.Security.Cryptography;
using System.Xml.Linq;
using System.Runtime.Intrinsics.Arm;
using Emgu.CV.Ocl;
using System.Threading.Channels;
using System.Diagnostics.Metrics;
using Emgu.CV.Reg;
using Emgu.CV.Stitching;
using Emgu.CV.Cuda;
using Emgu.CV.Features2D;
using Emgu.CV.XObjdetect;
using Emgu.CV.Face;
using Emgu.CV.XImgproc;

namespace ComputerVision
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Mat element = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(3, 3), new Point(-1, -1));
            Mat element2 = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1));
            Mat element3 = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(2, 2), new Point(-1, -1));
            float[,] matrix = new float[3, 3] {
                      { 1, 1, 1 },
                      {  1, -9, 1},
                      { 1, 1, 1 }
                    };
            ConvolutionKernelF sharpie = new ConvolutionKernelF(matrix);
            Mat longg = new Mat();
            Mat longDark = new Mat();
            Mat longThresh = new Mat();
            Mat longCanny = new Mat();
            Mat longLight = new Mat();
            Mat longResult = new Mat();
            Mat longDetails = new Mat();
            Mat long2 = new Mat();
            Mat long3 = new Mat();
            Mat longEdges = new Mat();
            Mat Blue = new Mat();
            Mat Green = new Mat();
            Mat Red = new Mat();

            //jeigu 5 mm arba daugiau = brokas

            string file = "Brown/1.bmp";
            //string file = "Rotated part/Brown/2.bmp";

            longg = CvInvoke.Imread(file, ImreadModes.Grayscale);
            long2 = CvInvoke.Imread(file);
            long3 = CvInvoke.Imread(file);


            //HaarDetectionType haar = new HaarDetectionType();

            string face2 = "shock";
            string ending2 = "png";
            string face = "hannah";
            string ending = "jpg";

            CascadeClassifier Classifier2 = new CascadeClassifier("haarcascades/haarcascade_frontalcatface_extended.xml");
            CascadeClassifier Classifier = new CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml");
            Image<Gray, byte> image = new Image<Gray, byte>("face/"+face+"."+ending);
            //Image<Bgr, byte> imageClean = new Image<Bgr, byte>("face.bmp");
            Mat imageClean = new Mat("face/"+face+"."+ending, ImreadModes.Grayscale);
            Image<Gray, byte> grayImage = image.Convert<Gray, byte>();
            Rectangle[] rectangles = Classifier.DetectMultiScale(grayImage, 1.2, 5, new Size(100,100),new Size(800,800));

            Image<Bgr, byte> destinationImage = new Image<Bgr, byte>("face/jackie.jpg");
            CvInvoke.ResizeForFrame(destinationImage, destinationImage, new Size(1000, 1000));

            Mat croppedImage = new Mat();


            foreach (Rectangle rectangle in rectangles)
            {
                int centerX = (rectangle.Left + rectangle.Right) / 2;
                int centerY = (rectangle.Top + rectangle.Bottom) / 2;
                int radius = rectangle.Width / 2;

                croppedImage = new Mat(imageClean, rectangle);

                // Create a mask for the filled circle
                Mat circleMask = new Mat(croppedImage.Size, DepthType.Cv8U, 1);
                circleMask.SetTo(new MCvScalar(255)); // Set the mask to all white
                CvInvoke.Circle(circleMask, new Point(croppedImage.Width / 2, croppedImage.Height / 2), radius, new MCvScalar(0), -1); // Fill the circle with black

                // Copy the filled circle into the region of interest (ROI)
                Mat roi = new Mat(imageClean, rectangle);
                Mat circleRegion = new Mat();

                CvInvoke.BitwiseAnd(croppedImage, circleMask, circleRegion);
                CvInvoke.BitwiseXor(circleRegion, croppedImage, circleRegion);
                circleRegion.CopyTo(roi);
                CvInvoke.Imshow("roi", roi);

            }

            Image<Gray, byte> image2 = new Image<Gray, byte>("face/" + face2 + "." + ending2);
            //Image<Bgr, byte> imageClean = new Image<Bgr, byte>("face.bmp");
            Mat imageClean2 = new Mat("face/" + face2 + "." + ending2, ImreadModes.Grayscale);
            Image<Gray, byte> grayImage2 = image2.Convert<Gray, byte>();
            Rectangle[] rectangles2 = Classifier2.DetectMultiScale(grayImage2, 1.2, 5, new Size(100, 100), new Size(800, 800));

            foreach (Rectangle rectangle in rectangles2)
            {
                int centerX = (rectangle.Left + rectangle.Right) / 2;
                int centerY = (rectangle.Top + rectangle.Bottom) / 2;
                int radius = rectangle.Width / 2;

                CvInvoke.ResizeForFrame(croppedImage, croppedImage, new Size(rectangle.Width, rectangle.Height));

                // Create a mask for the filled circle
                Mat circleMask = new Mat(croppedImage.Size, DepthType.Cv8U, 1);
                circleMask.SetTo(new MCvScalar(255)); // Set the mask to all white
                CvInvoke.Circle(circleMask, new Point(croppedImage.Width / 2, croppedImage.Height / 2), radius, new MCvScalar(0), -1); // Fill the circle with black

                // Copy the filled circle into the region of interest (ROI)
                Mat roi = new Mat(imageClean2, rectangle);
                Mat circleRegion = new Mat();
                CvInvoke.BitwiseAnd(croppedImage, circleMask, circleRegion);
                circleRegion.CopyTo(roi);
            }



            CvInvoke.ResizeForFrame(imageClean2, imageClean2, new Size(1000,1000));
            CvInvoke.Imshow("detected", imageClean2);
            CvInvoke.WaitKey();

            //CascadeClassifier
            //EigenFaceRecognizer
            //FisherFaceRecognizer
            //LocalBinarizationMethods


            /*//----------------------------------------------------------------------------------------------------------------------------------------------
            //shadow wizard money gang





            *//*Mat imag = new Mat("nums/nums1.bmp");
            Image<Gray, byte> gray = new Image<Gray, byte>("nums/nums1.bmp");
            Mat imagMatch = new Mat("find/1.bmp");
            Image<Gray, byte> grayMatch = new Image<Gray, byte>("find/1.bmp");
            //https://github.com/halanch599/EmguCV4.4/blob/master/EmgucvDemo/Form1.cs
            //https://stackoverflow.com/questions/53079167/emgucv-opencv-orbdetector-finding-only-bad-matches
            ORB detector2 = new ORB(650);
            FastFeatureDetector detector = new FastFeatureDetector();
            BFMatcher matcher = new BFMatcher(distanceType:Emgu.CV.Features2D.DistanceType.Hamming);
            var corners = detector2.Detect(gray);

            Mat outimg = new Mat();
            Features2DToolbox.DrawKeypoints(imag, new VectorOfKeyPoint(corners), outimg, new Bgr(0, 255, 0));
            CvInvoke.ResizeForFrame(outimg, outimg, new Size(1000, 1000));
            CvInvoke.Imshow("orb", outimg);
            CvInvoke.WaitKey();*/

            /*Stopwatch stopwatch = new Stopwatch();
            for (int i = 1; i < 6; i++)
            {
                stopwatch.Reset();
                stopwatch.Start();

                var model = new Mat("find/5_border_bigger.bmp");
                var scene = new Mat("nums/nums" + i + ".bmp", ImreadModes.Grayscale);

                var result = Draw(model, scene);
                CvInvoke.ResizeForFrame(result, result, new Size(1000, 1000));
                CvInvoke.Imshow("orb", result);
                stopwatch.Stop();
                Console.WriteLine("elapsed time " + stopwatch.ElapsedMilliseconds + "ms");
                CvInvoke.WaitKey();
            }*//*



            Stopwatch stopwatch = new Stopwatch();
            for (int i = 1; i < 6; i++)
            {
                for (int j = 1; j < 7; j++)
                {
                    stopwatch.Reset();
                    stopwatch.Start();

                    var model = new Mat("find2/"+j+".png", ImreadModes.Grayscale);
                    var scene = new Mat("nums/nums" + i + ".bmp", ImreadModes.Grayscale);
                    //CvInvoke.ResizeForFrame(scene, scene, new Size(1000,1000));
                    //CvInvoke.Threshold(scene, scene, 40, 255, ThresholdType.Binary);
                    CvInvoke.Threshold(model, model, 100, 255, ThresholdType.Otsu);
                    var result = Draw(model, scene);
                    CvInvoke.ResizeForFrame(result, result, new Size(1000, 1000));
                    CvInvoke.Imshow("orb " + j, result);
                    stopwatch.Stop();
                    Console.WriteLine("elapsed time " + stopwatch.ElapsedMilliseconds + " ms");
                }
                CvInvoke.WaitKey();
                Console.WriteLine(i + " done");
            }


            *//*Stopwatch stopwatch = new Stopwatch();
            for (int i = 1; i < 6; i++)
            {
                for (int j = 1; j < 6; j++)
                {
                    stopwatch.Reset();
                    stopwatch.Start();
                    var model = new Mat("find/"+j+".bmp", ImreadModes.Grayscale);
                    var scene = new Mat("nums/nums"+i+".bmp", ImreadModes.Grayscale);
                    CvInvoke.ResizeForFrame(model, model, new Size(1000, 1000));
                    //CvInvoke.AdaptiveThreshold(scene, scene, 255, AdaptiveThresholdType.GaussianC, ThresholdType.Binary, 17, 13);
                    CvInvoke.Threshold(scene, scene, 35, 255, ThresholdType.Binary);
                    CvInvoke.Threshold(model, model, 100, 255, ThresholdType.Otsu);
                    var result = Draw(model, scene);
                    CvInvoke.ResizeForFrame(result, result, new Size(1000, 1000));
                    CvInvoke.Imshow("orb " + j, result);
                    Console.WriteLine("elapsed time " + stopwatch.ElapsedMilliseconds + " ms");
                }
                CvInvoke.WaitKey();
                Console.WriteLine(i+ " done");
            }*//*


            //--------------------------------------------------------------------------------------------------------------------------------------------*/
            //Console.WriteLine(longg.NumberOfChannels);
            //Console.WriteLine(long2.NumberOfChannels);


            /*CvInvoke.Imwrite("longbw.bmp", longg);
            CvInvoke.Imwrite("longbgr.bmp", long2);

            CvInvoke.ResizeForFrame(longg, longg, new Size(1000, 1000));
            CvInvoke.ResizeForFrame(long2, long2, new Size(1000, 1000));
            CvInvoke.ResizeForFrame(long3, long3, new Size(1000, 1000));

            //CvInvoke.Rotate(longg, longg, RotateFlags.Rotate90CounterClockwise);
            //CvInvoke.Rotate(long2, long2, RotateFlags.Rotate90CounterClockwise);

            //Console.WriteLine(long2.NumberOfChannels);
            Mat[] channels = new Mat[3];
            channels = long2.Split();
            for (int i = 0; i < long2.NumberOfChannels; i++)
            {
                if (i == 0)
                {
                    Blue = channels[i];
                }
                if (i == 1)
                {
                    Green = channels[i];
                }
                if (i == 2)
                {
                    Red = channels[i];
                }
            }

            long2 = Blue;

            //100
            //6.5


            Console.WriteLine("Kiek mm skaitos klaida?(0-5000)");
            double limit = double.Parse(Console.ReadLine());
            Console.WriteLine("Koks min ratio tinkamas?(0-100)");
            double ratio = double.Parse(Console.ReadLine());
            double xkoef = 30;//63.846;
            double ykoef = 30;//40;

            //Console.WriteLine("Konvertavimo koeficientas?");
            //double koef = double.Parse(Console.ReadLine());


            CvInvoke.Imshow("base", long2);
            //CvInvoke.Imshow("base2", Green);
            //CvInvoke.WaitKey();


            //CvInvoke.AdaptiveThreshold(Blue, longThresh, 200, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 9, 0.5);    // Pabandyti sitaaa
            //CvInvoke.Imshow("adapt", longThresh);
            //CvInvoke.WaitKey();

            CvInvoke.Threshold(long2, longThresh, 20, 255, ThresholdType.BinaryInv);
            CvInvoke.Imshow("threshold inversion", longThresh);
            //CvInvoke.WaitKey();

            CvInvoke.Dilate(longThresh, longThresh, element3, new Point(-1, -1), 2, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
            CvInvoke.Imshow("dilate", longThresh);
            //CvInvoke.WaitKey(); 

            //CvInvoke.Erode(longThresh, longThresh, element3, new Point(-1, -1), 1, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
            //CvInvoke.Imshow("dilate", longThresh);
            //CvInvoke.WaitKey();

            CvInvoke.Threshold(long2, longResult, 35, 255, ThresholdType.Binary);
            CvInvoke.Imshow("threshold", longResult);
            //CvInvoke.WaitKey();

            CvInvoke.BitwiseXor(longResult, ~longThresh, longResult);
            CvInvoke.Imshow("bitwise", longResult);
            //CvInvoke.WaitKey();

            CvInvoke.Dilate(longResult, longResult, element3, new Point(-1, -1), 2, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
            CvInvoke.Imshow("dilate", longResult);
            //CvInvoke.WaitKey();

            CvInvoke.Threshold(Blue, Blue, 75, 255, ThresholdType.Binary);
            CvInvoke.Imshow("contour find", Blue);
            //CvInvoke.WaitKey();

            //longDetails = longResult;

            CvInvoke.BitwiseOr(longResult, Blue, Blue);
            CvInvoke.Imshow("minus", Blue);
            //CvInvoke.WaitKey();

            CvInvoke.Threshold(Red, Red, 2, 255, ThresholdType.BinaryInv);
            CvInvoke.Imshow("red", Red);
            //CvInvoke.WaitKey();



            CvInvoke.BitwiseAnd(long3, long3, Blue, Red);

            CvInvoke.Imshow("Result", Blue);
            //CvInvoke.WaitKey();



            //CvInvoke.CvtColor(longResult, longResult, ColorConversion.Bgr2Gray);
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(~longResult, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);
            GeneralContourFind(contours, longResult);
            int size = contours.Size;
            Rectangle[] array = new Rectangle[size];
            for (int i = 0; i < size; i++)
            {
                using (VectorOfPoint vectorOfPoint = contours[i])
                {
                    array[size - 1 - i] = CvInvoke.BoundingRectangle(vectorOfPoint);
                }
            }
            Mat longFresh = CvInvoke.Imread(file);
            CvInvoke.ResizeForFrame(longFresh, longFresh, new Size(1000, 1000));
            //CvInvoke.Rotate(longFresh, longFresh, RotateFlags.Rotate90CounterClockwise);





            LimitContourFind(contours, longFresh, limit, xkoef, ykoef, ratio);
            bool check = false;
            check = IsItDefective(contours, check, limit);
            Console.WriteLine("Is it defective : " + check);
            CvInvoke.Imshow("contoured long", longFresh);
            CvInvoke.WaitKey();
            CvInvoke.Imwrite("results.bmp", longFresh);
            //Console.WriteLine(CvInvoke.cvGetImageROI(longResult));





            Image<Bgr, Byte> img = new Image<Bgr, byte>("longbw.bmp");
            Image<Gray, Byte> imgbw = new Image<Gray, byte>("longbw.bmp");*/
            //HistImage(imgbw);

            Console.WriteLine("Hello World!");
        }































        public static Mat Draw(Mat modelImage, Mat observedImage)
        {
            Mat homography;
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
            {
                Mat mask;
                FindMatch(modelImage, observedImage, out modelKeyPoints, out observedKeyPoints, matches, out mask, out homography);
                Mat result = new Mat();
                Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
                    matches, result, new MCvScalar(255, 0, 0), new MCvScalar(0, 0, 255), mask);

                if (homography != null)
                {
                    var imgWarped = new Mat();
                    CvInvoke.WarpPerspective(observedImage, imgWarped, homography, modelImage.Size, Inter.Linear, Warp.InverseMap);
                    Rectangle rect = new Rectangle(Point.Empty, modelImage.Size);
                    var pts = new PointF[]
                    {
                  new PointF(rect.Left, rect.Bottom),
                  new PointF(rect.Right, rect.Bottom),
                  new PointF(rect.Right, rect.Top),
                  new PointF(rect.Left, rect.Top)
                    };

                    pts = CvInvoke.PerspectiveTransform(pts, homography);
                    var points = new Point[pts.Length];
                    for (int i = 0; i < points.Length; i++)
                        points[i] = Point.Round(pts[i]);

                    using (var vp = new VectorOfPoint(points))
                    {
                        CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                    }
                }
                return result;
            }
        }
        public static void FindMatch(Mat modelImage, Mat observedImage, out VectorOfKeyPoint modelKeyPoints, out VectorOfKeyPoint observedKeyPoints, VectorOfVectorOfDMatch matches, out Mat mask, out Mat homography)
        {
            int k = 2;
            double uniquenessThreshold = 0.90;
            homography = null;
            modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();
            using (UMat uModelImage = modelImage.GetUMat(AccessType.Read))
            using (UMat uObservedImage = observedImage.GetUMat(AccessType.Read))
            {
                var featureDetector = new ORB(650);
                Mat modelDescriptors = new Mat();
                featureDetector.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);
                Mat observedDescriptors = new Mat();
                featureDetector.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);
                //https://dsp.stackexchange.com/questions/28557/norm-hamming2-vs-norm-hamming
                //hamming2 mystery
                using (var matcher = new BFMatcher(distanceType: Emgu.CV.Features2D.DistanceType.Hamming2, false))
                {
                    matcher.Add(modelDescriptors);

                    matcher.KnnMatch(observedDescriptors, matches, k, null);
                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                    int nonZeroCount = CvInvoke.CountNonZero(mask);
                    if (nonZeroCount >= 4)
                    {
                        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                            matches, mask, 1.5, 20);
                        if (nonZeroCount >= 4)
                            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints,
                                observedKeyPoints, matches, mask, 2);
                    }
                }
            }
        }

        public static void HistImage(Image<Gray, Byte> img)
        {
            DenseHistogram Histo = new DenseHistogram(255, new RangeF(0, 255));

            Image<Gray, Byte>[] channels = img.Split();


            for (int j = 0; j < channels.Length; j++)
            {

                Histo.Calculate(new Image<Gray, Byte>[] { channels[j] }, true, null);
                Mat histImage = new Mat();
                int width = 255, height = 255;
                Mat histImg = new Mat(new Size(width, height), DepthType.Cv8U, 3);
                histImg.SetTo(new MCvScalar(255, 255, 255));
                float[] floatHist = Histo.GetBinValues();
                for (int i = 0; i < 255; i++)
                {
                    if (i < 21)
                    {
                        floatHist[i] = 0;
                        Point p1 = new Point(i, histImg.Rows - (int)floatHist[i]);
                        Point p2 = new Point(i, histImg.Rows);
                        CvInvoke.Line(histImg, p1, p2, new MCvScalar(0, 0, 0), 1);
                    }
                }
                long max = (long)floatHist.Max();
                for (int i = 0; i < 255; i++)
                {
                    floatHist[i] = (float)(floatHist[i] / max * histImg.Rows);
                    Point p1 = new Point(i, histImg.Rows - (int)floatHist[i]);
                    Point p2 = new Point(i, histImg.Rows);
                    CvInvoke.Line(histImg, p1, p2, new MCvScalar(0, 0, 0), 1);
                }

                CvInvoke.Imshow("Histogram ch" + j, histImg);
                CvInvoke.WaitKey();

            }
        }
        public static void FindCircles(VectorOfVectorOfPoint contours, Mat circleMat, Mat mat)
        {
            int size = contours.Size;
            Rectangle[] arrayFull = new Rectangle[size];
            CircleF[] circles = new CircleF[size];
            int count = 0;
            for (int i = 0; i < size; i++)
            {
                using (VectorOfPoint vectorOfPoint = contours[i])
                {
                    arrayFull[size - 1 - i] = CvInvoke.BoundingRectangle(vectorOfPoint);
                }
            }
            for (int i = 0; i < size; i++)
            {
                double perimeter = CvInvoke.ArcLength(contours[i], true);
                VectorOfPoint approx = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contours[i], approx, 0.04 * perimeter, true);
                //CvInvoke.DrawContours(mat, contours, i, new MCvScalar(100), 2);

                var moments = CvInvoke.Moments(contours[i]);
                int x = (int)(moments.M10 / moments.M00);
                int y = (int)(moments.M01 / moments.M00);

                double radius = Math.Sqrt(CvInvoke.ContourArea(contours[i]) / Math.PI);


                if (approx.Size >= 5)
                {
                    CircleF circle = new CircleF(new Point(x, y), (int)radius);
                    if (circle.Center.IsEmpty == false)
                    {
                        circles[count] = circle;
                        count++;
                        CvInvoke.Circle(mat, new Point(x, y), (int)radius, new MCvScalar(100), 2);
                        CvInvoke.Circle(circleMat, new Point(x, y), (int)radius, new MCvScalar(100), 2);
                        CvInvoke.FloodFill(circleMat, new Mat(), new Point((int)circle.Center.X, (int)circle.Center.Y), new MCvScalar(255), out _, new MCvScalar(0), new MCvScalar(0));
                        CvInvoke.PutText(mat, "circle", new Point(x, y), FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 1);
                    }

                }
            }
            CvInvoke.Imshow("results", mat);
            CvInvoke.WaitKey();
        }
        public static void Ausytes(VectorOfVectorOfPoint otherContours, VectorOfVectorOfPoint contours, Mat mat, Rectangle[] array)
        {
            for (int i = 0; i < otherContours.Size; i++)
            {
                if (otherContours[i].Size > 4)
                {
                    CvInvoke.DrawContours(mat, otherContours, i, new MCvScalar(100), 2);
                    using (VectorOfPoint vectorOfPoint = contours[i])
                    {
                        var moments = CvInvoke.Moments(otherContours[i]);
                        int x = (int)(moments.M10 / moments.M00);
                        int y = (int)(moments.M01 / moments.M00);
                        array[contours.Size - 1 - i] = CvInvoke.BoundingRectangle(vectorOfPoint);
                        CvInvoke.PutText(mat, "thing", new Point(x, y), FontFace.HersheySimplex, 0.5, new MCvScalar(0, 0, 255), 1);
                    }
                }

            }

            CvInvoke.Imshow("results next", mat);
            CvInvoke.WaitKey();
        }
        public static void GeneralContourFind(VectorOfVectorOfPoint otherContours, Mat mat)
        {
            for (int i = 0; i < otherContours.Size; i++)
            {
                var area = CvInvoke.ContourArea(otherContours[i]);
                if (area > 700 && area < 5000)
                {
                    CvInvoke.DrawContours(mat, otherContours, i, new MCvScalar(0, 255, 0), 1);
                }
            }
            //CvInvoke.Imshow("results next", mat);
            //CvInvoke.WaitKey();
        }
        /* public static void LimitContourFind(VectorOfVectorOfPoint otherContours, Mat mat, double limit)
         {
             for (int i = 0; i < otherContours.Size; i++)
             {
                 var area = CvInvoke.ContourArea(otherContours[i]);
                 if (area > limit && area < 5000)
                 {
                     CvInvoke.DrawContours(mat, otherContours, i, new MCvScalar(0, 255, 0), 1);
                 }
             }
         } */
        public static void LimitContourFind(VectorOfVectorOfPoint otherContours, Mat mat, double limit, double xkoef, double ykoef, double ratio)
        {
            for (int i = 0; i < otherContours.Size; i++)
            {
                Rectangle rect = CvInvoke.BoundingRectangle(otherContours[i]);

                double contourArea = CvInvoke.ContourArea(otherContours[i]);
                double boundingRectArea = rect.Width * rect.Height;

                double areaRatio = contourArea / boundingRectArea * 100;

                double horizontalDimensionMm = rect.Width * xkoef;
                double verticalDimensionMm = rect.Height * ykoef;


                if (horizontalDimensionMm > limit && verticalDimensionMm > limit && horizontalDimensionMm < 5000 && verticalDimensionMm < 5000 && areaRatio > ratio)
                {
                    CvInvoke.DrawContours(mat, otherContours, i, new MCvScalar(0, 255, 0), 1);
                }
            }
        }
        public static bool IsItDefective(VectorOfVectorOfPoint otherContours, bool check, double limit)
        {
            for (int i = 0; i < otherContours.Size; i++)
            {
                var area = CvInvoke.ContourArea(otherContours[i]);
                if (area > limit && area < 5000)
                {
                    check = true;
                    return check;
                }
            }
            return check;
            //CvInvoke.Imshow("results next", mat);
            //CvInvoke.WaitKey();
        }
        public static List<Mat> PrintParts(VectorOfVectorOfPoint contours, Mat mat, List<Mat> resultList)
        {
            int size = contours.Size;
            Rectangle[] arrayFull = new Rectangle[size];
            for (int i = 0; i < size; i++)
            {
                using (VectorOfPoint vectorOfPoint = contours[i])
                {
                    arrayFull[size - 1 - i] = CvInvoke.BoundingRectangle(vectorOfPoint);
                }
            }

            for (int i = 0; i < arrayFull.Length; i++)
            {
                Mat result = new Mat(mat, arrayFull[i]).Clone();
                resultList.Add(result);
                CvInvoke.Imwrite("result " + i + ".bmp", result);
                CvInvoke.Imshow("reshaped??", resultList[i]);
                CvInvoke.WaitKey();
            }
            return resultList;
        }
    }
}


//savartynas


/*Mat element = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(3, 3), new Point(-1, -1));
float[,] matrix = new float[3, 3] {
                      { 0, -1, 0 },
                      {  -1, 5, -1},
                      { 0, -1, 0 }
                    };
ConvolutionKernelF sharpie = new ConvolutionKernelF(matrix);
Mat element2 = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1));
Mat element3 = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(2, 2), new Point(-1, -1));
Mat longg = new Mat();
Mat longDark = new Mat();
Mat longThresh = new Mat();
Mat longCanny = new Mat();
Mat longLight = new Mat();
Mat longResult = new Mat();
Mat longDetails = new Mat();
Mat long2 = new Mat();

longg = CvInvoke.Imread("long.bmp", ImreadModes.Grayscale);
long2 = CvInvoke.Imread("long.bmp");
*//*            Rectangle rectangle = new Rectangle(0, 0, 512, 1200);
            Mat longg = new Mat(longgt, rectangle);
            Mat long2 = new Mat(long2t, rectangle);*//*
CvInvoke.Imwrite("longbw.bmp", longg);
CvInvoke.Imwrite("longbgr.bmp", long2);

CvInvoke.Rotate(longg, longg, RotateFlags.Rotate90CounterClockwise);
CvInvoke.Rotate(long2, long2, RotateFlags.Rotate90CounterClockwise);

CvInvoke.Threshold(longg, longDetails, 170, 255, ThresholdType.ToZeroInv);
//CvInvoke.Imshow("threshold Details", longDetails);
//CvInvoke.WaitKey();

//CvInvoke.Threshold(longg, longDark, 200, 255, ThresholdType.Binary);
//CvInvoke.Imshow("threshold longDark", longDark);
////CvInvoke.WaitKey();

//CvInvoke.Threshold(~longg, longLight, 20, 255, ThresholdType.BinaryInv);
//CvInvoke.Dilate(longLight, longLight, element, new Point(-1, -1), 2, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
//CvInvoke.Imshow("threshold longLight", longLight);
//CvInvoke.WaitKey();

CvInvoke.Threshold(longg, longThresh, 185, 255, ThresholdType.ToZeroInv);
//CvInvoke.Canny(longThresh, longCanny, 170, 185);
//CvInvoke.Imshow("Canny", longCanny);
CvInvoke.Imshow("threshold longThresh before", longThresh);
//CvInvoke.Filter2D(longThresh, longThresh, sharpie, new Point(0, 0)); // sitas cool tho
//CvInvoke.GaussianBlur(longThresh, longThresh, new Size(5, 5), 2);
CvInvoke.Imshow("threshold longThresh", longThresh);
//CvInvoke.WaitKey();

CvInvoke.Erode(longThresh, longResult, element3, new Point(-1, -1), 7, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
//CvInvoke.BitwiseXor(longThresh, longResult, longResult);

//CvInvoke.Erode(longResult, longResult, element, new Point(-1, -1), 2, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
//CvInvoke.Dilate(longResult, longResult, element, new Point(-1, -1), 2, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
CvInvoke.Imshow("threshold longThresher", longResult);
//CvInvoke.WaitKey();

//CvInvoke.BitwiseAnd(longDark, longLight, longResult);
//longResult = ~(longDark | longLight);
//CvInvoke.Dilate(longResult, longResult, element, new Point(-1, -1), 2, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));*/

/*          CvInvoke.Threshold(longg, longDark, 20, 255, ThresholdType.Binary);
                   CvInvoke.Dilate(longDark, longDark, element, new Point(-1, -1), 1, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
                   CvInvoke.Imshow("threshold longDark", longDark);
                   CvInvoke.WaitKey();
                   CvInvoke.Threshold(~longg, longLight, 145, 255, ThresholdType.Binary);
                   CvInvoke.Imshow("threshold longLight", longLight);
                   CvInvoke.WaitKey();
                   CvInvoke.BitwiseXor(longDark, longLight, longResult);
                   CvInvoke.Dilate(longResult, longResult, element, new Point(-1, -1), 2, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
                   CvInvoke.Imshow("longResult", longResult);
                   CvInvoke.WaitKey();*/




/*Mat mat = new Mat();
Mat element = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(3, 3), new Point(-1, -1));
mat = CvInvoke.Imread("uzd21.bmp", ImreadModes.Grayscale);
CvInvoke.ResizeForFrame(mat, mat, new Size(1000, 1000));
CvInvoke.Imshow("read", mat);
CvInvoke.WaitKey();
Mat thresholded = new Mat();
CvInvoke.Threshold(mat, thresholded, 240, 255, ThresholdType.Binary);
CvInvoke.Dilate(thresholded, thresholded, element, new Point(-1, -1), 1, BorderType.Reflect, new MCvScalar(255, 255, 255));
CvInvoke.Erode(thresholded, thresholded, element, new Point(-1, -1), 3, BorderType.Reflect, new MCvScalar(255, 255, 255));
CvInvoke.Dilate(thresholded, thresholded, element, new Point(-1, -1), 3, BorderType.Reflect, new MCvScalar(255, 255, 255));
CvInvoke.Imshow("thresholded", thresholded);
CvInvoke.WaitKey();


Console.WriteLine(thresholded.NumberOfChannels);
VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
CvInvoke.FindContours(thresholded, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

FindCircles(contours, thresholded, mat);

CvInvoke.Threshold(thresholded, thresholded, 40, 255, ThresholdType.Binary);
CvInvoke.Dilate(thresholded, thresholded, element, new Point(-1, -1), 1, BorderType.Reflect, new MCvScalar(255, 255, 255));


VectorOfVectorOfPoint otherContours = new VectorOfVectorOfPoint();
Rectangle[] array = new Rectangle[contours.Size];
CvInvoke.FindContours(thresholded, otherContours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

Ausytes(otherContours, contours, mat, array);

//List<Mat> resultList = new List<Mat>();

//resultList = PrintParts(contours, mat, resultList);
//for (int i = 0; i < resultList.Count; i++)
//{
//    Image<Bgr, Byte> img = new Image<Bgr, byte>("result " + i + ".bmp");

//    HistImage(img);
//}*/





//CvInvoke.Erode(longResult, longResult, element3, new Point(-1, -1), 1, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
//CvInvoke.Imshow("Erode", longResult);
//CvInvoke.WaitKey();

/*CvInvoke.EdgePreservingFilter(longg, longThresh, EdgePreservingFilterFlag.RecursFilter);
CvInvoke.Imshow("edges", longThresh);
CvInvoke.WaitKey();

CvInvoke.FilterSpeckles(longThresh, 10, 3, 30);
CvInvoke.Imshow("specles", longThresh);
CvInvoke.WaitKey();*/

//CvInvoke.AdaptiveThreshold(longg, longThresh, 200, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 9 , 0.5);    // Pabandyti sitaaa
//CvInvoke.Imshow("adapt", longThresh);
//CvInvoke.WaitKey();

//CvInvoke.Dilate(longThresh, longThresh, element, new Point(-1, -1), 1, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
//CvInvoke.Imshow("dilate", longThresh);
//CvInvoke.WaitKey();

/*CvInvoke.Dilate(longThresh, longThresh, element3, new Point(-1, -1), 1, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
CvInvoke.Imshow("dilate", longThresh);
CvInvoke.WaitKey();

CvInvoke.Dilate(longThresh, longThresh, element3, new Point(-1, -1), 1, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
CvInvoke.Imshow("dilate", longThresh);
CvInvoke.WaitKey();

CvInvoke.Erode(longThresh, longThresh, element, new Point(-1, -1), 2, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
CvInvoke.Imshow("erode", longThresh);
CvInvoke.WaitKey();

CvInvoke.Threshold(longThresh, longResult, 108, 255, ThresholdType.Binary);
CvInvoke.Imshow("longResult threshold", longResult);
CvInvoke.WaitKey();*/

//CvInvoke.BitwiseXor(long2, longThresh, longResult);
//CvInvoke.Imshow("bitwise", longResult);
//CvInvoke.WaitKey();

//CvInvoke.BitwiseXor(longThresh, longResult, longResult);
//CvInvoke.Imshow("bitwise result2", longResult);
//CvInvoke.WaitKey();

//CvInvoke.Dilate(longResult, longResult, element, new Point(-1, -1), 1, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
//CvInvoke.Imshow("dilate", longResult);
//CvInvoke.WaitKey();

//CvInvoke.Erode(longResult, longResult, element3, new Point(-1, -1), 1, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));
//CvInvoke.Imshow("erode", longResult);
//CvInvoke.WaitKey();

//CvInvoke.Threshold(~longg, longDetails, 120, 255, ThresholdType.ToZeroInv);
//CvInvoke.Imshow("threshold longDetails", longDetails);
//CvInvoke.WaitKey();
//CvInvoke.Threshold(~longg, longThresh, 185, 255, ThresholdType.ToZeroInv);
//CvInvoke.Canny(longg, longCanny, 170, 185);
//CvInvoke.Imshow("Canny", longCanny);
//longCanny = longThresh;
//CvInvoke.Imshow("threshold longThresh", longThresh);
//CvInvoke.WaitKey();
//CvInvoke.Filter2D(longCanny, longCanny, sharpie, new Point(0, 0)); // sitas cool tho
//CvInvoke.GaussianBlur(longThresh, longThresh, new Size(5, 5), 2);
//CvInvoke.Imshow("threshold longThresh after", longCanny);
//CvInvoke.WaitKey();

/*            CvInvoke.BitwiseXor(longThresh, longDetails, longResult);
            CvInvoke.Imshow("bitten", longThresh);
            CvInvoke.WaitKey();*/

//CvInvoke.Dilate(longThresh, longResult, element3, new Point(-1, -1), 3, BorderType.Reflect, new Emgu.CV.Structure.MCvScalar(255, 255, 255));

//CvInvoke.Imshow("threshold longResult before", longResult);
//CvInvoke.WaitKey();
//longResult = longDetails;
//CvInvoke.Threshold(longResult, longResult, 19, 255, ThresholdType.Binary);
//CvInvoke.Imshow("longResult", longResult);
//CvInvoke.WaitKey();






//List<Rectangle> mergedRectangles = new List<Rectangle>();
/*bool nothinghappened = false;
while (!nothinghappened)
{
    nothinghappened = true;
    for (int i = 0; i < array.Count(); i++)
    {
        for (int j = 0; j < array.Count(); j++)
        {
            if (i == j) continue;
            Rectangle rect = array[i];
            Rectangle rect2 = array[j];

            double centerX1 = rect.X + rect.Width / 2.0;
            double centerY1 = rect.Y + rect.Height / 2.0;
            double centerX2 = rect2.X + rect2.Width / 2.0;
            double centerY2 = rect2.Y + rect2.Height / 2.0;
            double distance = Math.Sqrt(Math.Pow(centerX2 - centerX1, 2) + Math.Pow(centerY2 - centerY1, 2));

            if ((rect.Width < 500 && rect2.Width < 500 && rect != rect2 && rect.IntersectsWith(rect2)) || (distance < 30 && rect.Width < 500 && rect2.Width < 500 && rect != rect2))
            {
                Rectangle merged = Rectangle.Union(rect, rect2);
                array[i] = merged;
                array[j] = Rectangle.Empty;
                nothinghappened = false;
                break;
            }
            if (rect.Width > 1000)
            {
                array[i] = Rectangle.Empty;
                nothinghappened = false;
                break;
            }
        }
        if (!nothinghappened)
            break;
    }
}
foreach (var item in array)
{
    if (item.Width > 5 && item.Height > 5)
    {
        mergedRectangles.Add(item);
    }
}*/
/*for (int i = 0; i < array.Count(); i++)
{   
    Rectangle rect = array[i];
    if (rect.Width < 500)
    {
        mergedRectangles.Add(rect);
    }

}

foreach (Rectangle rect in mergedRectangles)
{
    CvInvoke.Rectangle(longFresh, rect, new MCvScalar(0, 255, 0), 2);
}*/
//for (int i = 0; i < contours.Size; i++)
//{
//    CvInvoke.DrawContours(longFresh, contours[i], i+1, new MCvScalar(0,255,0), 2);
//}