package br.com.helpdev.facedetect;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.maedi.soft.ino.base.BuildActivity;
import com.maedi.soft.ino.base.func_interface.ActivityListener;
import com.maedi.soft.ino.base.store.MapDataParcelable;
import com.maedi.soft.ino.base.utils.ScreenSize;
import com.maedi.soft.ino.recognize.face.R;
import com.maedi.soft.ino.recognize.face.Recognizer;
import com.maedi.soft.ino.recognize.face.env.FileUtils;
import com.maedi.soft.ino.recognize.face.env.ImageUtils;
import com.maedi.soft.ino.recognize.face.ml.BlazeFace;
import com.maedi.soft.ino.recognize.face.ml.FaceNet;
import com.maedi.soft.ino.recognize.face.ml.LibSVM;
import com.maedi.soft.ino.recognize.face.recognizeupdate.SupportVectorMachine;
import com.maedi.soft.ino.recognize.face.tracking.MultiBoxTracker;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.LinkedList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;
import butterknife.BindView;
import butterknife.ButterKnife;
import pub.devrel.easypermissions.EasyPermissions;
import timber.log.Timber;

public class ScanRecognizeFace extends BuildActivity<View> implements ActivityListener<Integer>, CameraBridgeViewBase.CvCameraViewListener2, EasyPermissions.PermissionCallbacks {

    private final String TAG = this.getClass().getName() +"- ScanRecognizeFace - ";

    private FragmentActivity f;

    @BindView(R.id.user_name)
    TextView textUserName;

    @BindView(R.id.tv)
    TextView infoFaces;

    @BindView(R.id.count_time)
    TextView textCountTime;

    @BindView(R.id.layout_right)
    LinearLayout layoutRight;

    //@BindView(R.id.main_surface)
    CameraBridgeViewBase cameraBridgeViewBase;

    private static final Scalar FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            Timber.d(TAG+"onManagerConnected Status -> "+status +" | "+ LoaderCallbackInterface.SUCCESS);
            if (status == LoaderCallbackInterface.SUCCESS) {
                Timber.d(TAG+"OpenCV loaded successfully");
                cameraBridgeViewBase.enableView();
            } else {
                super.onManagerConnected(status);
            }
        }
    };

    private static final int CROP_HEIGHT = BlazeFace.INPUT_SIZE_HEIGHT;
    private static final int CROP_WIDTH = BlazeFace.INPUT_SIZE_WIDTH;
    private Integer sensorOrientation;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    //private volatile boolean running = false;
    private volatile int qtdFaces;
    private volatile Mat matTmpProcessingFace, matTmpProcessingRgbaFace;
    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CascadeClassifier cascadeClassifier;
    private File mCascadeFile;
    private Recognizer recognizer;

    private boolean waitSeconds;
    private Timer timer;
    private TimerTask timerTask;
    private int counterTimer;

    @Override
    public int setPermission() {
        return 0;
    }

    @Override
    public boolean setAnalytics() {
        return false;
    }

    @Override
    public int baseContentView() {
        return R.layout.activity_scan_face;
    }

    @Override
    public ActivityListener createListenerForActivity() {
        return this;
    }

    @Override
    public void onCreateActivity(Bundle savedInstanceState) {
        f = this;
        ButterKnife.bind(this);

        cameraBridgeViewBase = findViewById(R.id.main_surface);

        waitSeconds = false;
        counterTimer = 7;

        RelativeLayout.LayoutParams layoutParams = new RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.MATCH_PARENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
        layoutParams.width = ScreenSize.instance(f).getWidth()/2;
        cameraBridgeViewBase.setLayoutParams(layoutParams);

        LinearLayout.LayoutParams layoutParams2 = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        layoutParams2.width = ScreenSize.instance(f).getWidth()/2;
        layoutParams2.gravity = Gravity.CENTER;
        layoutRight.setGravity(Gravity.CENTER);
        layoutRight.setLayoutParams(layoutParams2);
        int rc = ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA);
        if (rc != PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "PERMISSION_REQUIRED");
            checkPermissions();
        }
        else
        {
            loadCameraBridge();
            setInit();
        }
    }

    @Override
    public void onBuildActivityCreated() {

    }

    private void setInit()
    {
        init();
        loadHaarCascadeFile();
        resumeOCV();
    }

    private void init()
    {
        File dir = new File(FileUtils.ROOT);
        if (dir.isDirectory()) {
            if (dir.exists()) dir.delete();
            dir.mkdirs();

            AssetManager mgr = getAssets();
            FileUtils.copyAsset(mgr, FileUtils.DATA_FILE);
            FileUtils.copyAsset(mgr, FileUtils.MODEL_FILE);
            FileUtils.copyAsset(mgr, FileUtils.LABEL_FILE);
        }

        try {
            blazeFace = BlazeFace.create(getAssets());
            faceNet = FaceNet.create(getAssets());
            svm = LibSVM.getInstance();
            recognizer = Recognizer.getInstance(getAssets());
        } catch (Exception e) {
            Log.d(TAG, "Exception initializing classifier! -> "+e.getMessage());
            finish();
        }
    }

    private void checkPermissions() {
        Log.d(TAG, "Camera permission is not granted. Requesting permission");
        if (hasAPI_LEVEL24_ANDROID_7_Above())
            CameraPermission_API_LEVEL24_ANDROID_7_Above(f);
        else CameraPermission(f);
        //if (isPermissionGranted()) {
        //    loadCameraBridge();
        //} else {
        //    ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        //}
    }

    private boolean isPermissionGranted() {
        return ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    public final int PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6 = 17;
    public final String[] GALLERY_PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    public void CameraPermission_API_LEVEL24_ANDROID_7_Above(FragmentActivity f){

        EasyPermissions.requestPermissions(f, "ALLOW CAMERA",
                PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6, GALLERY_PERMISSIONS);
    }

    public void CameraPermission(FragmentActivity f) {
        if (ContextCompat.checkSelfPermission(f, GALLERY_PERMISSIONS[0]) != PackageManager.PERMISSION_GRANTED) {

            if (ActivityCompat.shouldShowRequestPermissionRationale(f, GALLERY_PERMISSIONS[0])) {

                ActivityCompat.requestPermissions(f,
                        GALLERY_PERMISSIONS,
                        PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6);

            } else {

                ActivityCompat.requestPermissions(f,
                        GALLERY_PERMISSIONS,
                        PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6);
            }
        }
    }

    public boolean hasAPI_LEVEL24_ANDROID_7_Above() {
        return Build.VERSION.SDK_INT >= 24; //API Level 7
    }

    private void loadCameraBridge() {
        Timber.d(TAG+"LOAD_OPEN_CAMERA");
        cameraBridgeViewBase.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
    }

    private void loadHaarCascadeFile() {
        Timber.d(TAG+"loadHaarCascadeFile -> called");
        try {
            File cascadeDir = getDir("haarcascade_frontalface_alt", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");

            if (!mCascadeFile.exists()) {
                FileOutputStream os = new FileOutputStream(mCascadeFile);
                InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
                is.close();
                os.close();
            }
        } catch (Throwable throwable) {
            Timber.d(TAG+"loadHaarCascadeFile exception -> "+throwable.getMessage());
            throw new RuntimeException("Failed to load Haar Cascade file");
        }
    }

    private void resumeOCV() {
        Timber.d(TAG+"resumeOCV -> called");
        if (OpenCVLoader.initDebug()) {
            Timber.d(TAG+"OpenCV library found inside package. Using it!");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Timber.d(TAG+"Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        }
        cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        cascadeClassifier.load(mCascadeFile.getAbsolutePath());
        //startFaceDetect();
    }

    private void setMinFaceSize(float faceSize) {
        mRelativeFaceSize = faceSize;
        mAbsoluteFaceSize = 0;
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //Log.d(TAG, "OnCameraFrame");
        //if (matTmpProcessingFace == null) {
        matTmpProcessingFace = inputFrame.gray();
        //}
        matTmpProcessingRgbaFace = inputFrame.rgba();
        if(!waitSeconds) {

            if (mAbsoluteFaceSize == 0) {
                int height = matTmpProcessingFace.rows();
                if (Math.round(height * mRelativeFaceSize) > 0) {
                    mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
                }
            }

            MatOfRect matOfRect = new MatOfRect();
            //cascadeClassifier.detectMultiScale(matTmpProcessingFace, matOfRect);
            cascadeClassifier.detectMultiScale(matTmpProcessingFace, matOfRect, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
            //org.opencv.core.Rect[] nrect = matOfRect.toArray();
            //String strRect = nrect.toString();
            //Log.d("RECT", strRect);
            //Log.d(TAG, "matTmpProcessingRgbaFace:"+matTmpProcessingRgbaFace);
            int newQtdFaces = matOfRect.toList().size();
            //Log.d(TAG, "FACE_COUNT:"+newQtdFaces +" | "+ qtdFaces);
            if (qtdFaces != newQtdFaces) {
                qtdFaces = newQtdFaces;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        infoFaces.setText(String.format(getString(R.string.faces_detects), qtdFaces));
                    }
                });
            }

            org.opencv.core.Rect[] facesArray = matOfRect.toArray();
            //detect if face == 1
            if (newQtdFaces == 1) {
                //create bitmap image
                Mat nwmath = new Mat();
                org.opencv.core.Rect nwr = facesArray[0];
                nwmath = matTmpProcessingRgbaFace.submat(nwr);
                Bitmap bitmap = Bitmap.createBitmap(nwmath.width(), nwmath.height(), Bitmap.Config.ARGB_8888);
                Bitmap bmpGrayScale = getGrayscale(bitmap);
                Utils.matToBitmap(nwmath, bmpGrayScale);

                sensorOrientation = (int)cameraBridgeViewBase.getRotation() - getScreenOrientation();
                frameToCropTransform =
                        ImageUtils.getTransformationMatrix(
                                nwmath.width(), nwmath.height(),
                                CROP_WIDTH, CROP_HEIGHT,
                                sensorOrientation, false);

                cropToFrameTransform = new Matrix();
                frameToCropTransform.invert(cropToFrameTransform);

                try {
                    boolean isEmptyImage = scanData(1, bmpGrayScale);
                    Timber.d(TAG+"isEmptyImage:"+isEmptyImage);
                    if(!isEmptyImage)
                    {
                       runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                waitSeconds = true;
                                //recognizeImage(bmpGrayScale, cropToFrameTransform);
                                SupportVectorMachine svm = new SupportVectorMachine(f, SupportVectorMachine.RECOGNITION);
                                Mat mat = new Mat();
                                Utils.bitmapToMat(bmpGrayScale, mat);
                                String labelName = svm.recognize(mat, "test");
                                //String labelName = svm.setRecognizeProbability(mat);
                                Timber.d(TAG+"LABEL_NAME -> "+labelName);
                                Toast.makeText(f, "Success Recognize > Label Name > "+labelName, Toast.LENGTH_SHORT).show();
                                textCountTime.setVisibility(View.VISIBLE);
                                startTimer();
                            }
                        });
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    Log.d(TAG, "Exception:"+e.getMessage());
                }
            }

            //Log.d(TAG, "FACE_ARRAY_COUNT:"+facesArray.length);
            for (int i = 0; i < facesArray.length; i++) {
                Imgproc.rectangle(matTmpProcessingRgbaFace, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
            }
        }

        return matTmpProcessingRgbaFace;
    }

    boolean findDifference(Bitmap firstImage, Bitmap secondImage)
    {
        int threashold = 10;
        if (firstImage.getHeight() != secondImage.getHeight() || firstImage.getWidth() != secondImage.getWidth())
            Toast.makeText(this, "Images size are not same", Toast.LENGTH_LONG).show();

        boolean isSame = true;

        for (int i = 0; i < firstImage.getWidth(); i++)
        {
            for (int j = 0; j < firstImage.getHeight(); j++)
            {
                int pixel = firstImage.getPixel(i,j);
                int redValue = Color.red(pixel);
                int blueValue = Color.blue(pixel);
                int greenValue = Color.green(pixel);

                int pixel2 = secondImage.getPixel(i,j);
                int redValue2 = Color.red(pixel2);
                int blueValue2 = Color.blue(pixel2);
                int greenValue2 = Color.green(pixel2);

                if (Math.abs(redValue2 - redValue) + Math.abs(blueValue2 - blueValue) + Math.abs(greenValue2 - greenValue) <= threashold)
//                if (firstImage.getPixel(i,j) == secondImage.getPixel(i,j))
                {
                }
                else
                {
                    secondImage.setPixel(i,j, Color.YELLOW); //for now just changing difference to yello color
                    isSame = false;
                }
            }
        }

        return isSame;
    }

    List<ScanRecognizeFace.Recognition> recognizeImage(Bitmap bitmap, Matrix matrix) {
        synchronized (this) {
            List<RectF> faces = blazeFace.detect(bitmap);
            final List<ScanRecognizeFace.Recognition> mappedRecognitions = new LinkedList<>();

            for (RectF rectF : faces) {
                Rect rect = new Rect();
                rectF.round(rect);

                FloatBuffer buffer = faceNet.getEmbeddings(bitmap, rect);
                LibSVM.Prediction prediction = svm.predict(buffer);

                matrix.mapRect(rectF);
                int index = prediction.getIndex();
                Timber.d(TAG + "===recognizer.classNames=== " + recognizer.classNames);
                Timber.d(TAG + "===recognizer.IndexIs=== " + index +" , classNames.size= "+ recognizer.classNames.size());
                if(null != recognizer.classNames) {
                    if(recognizer.classNames.size() > 0) {
                        if(index < recognizer.classNames.size()) {
                            String name = recognizer.classNames.get(index);
                            Timber.d(TAG + "DATA_RESULT_NAME - " + name);
                            ScanRecognizeFace.Recognition result = new ScanRecognizeFace.Recognition("" + index, name, prediction.getProb(), rectF);
                            Timber.d(TAG + "DATA_RESULT_PERSON - " + result);
                            mappedRecognitions.add(result);
                        }
                    }
                }
            }
            return mappedRecognitions;
        }
    }

    private Bitmap getGrayscale(Bitmap src){

        //Custom color matrix to convert to GrayScale
        float[] matrix = new float[]{
                0.3f, 0.59f, 0.11f, 0, 0,
                0.3f, 0.59f, 0.11f, 0, 0,
                0.3f, 0.59f, 0.11f, 0, 0,
                0, 0, 0, 1, 0,};

        Bitmap dest = Bitmap.createBitmap(
                src.getWidth(),
                src.getHeight(),
                src.getConfig());

        Canvas canvas = new Canvas(dest);
        Paint paint = new Paint();
        ColorMatrixColorFilter filter = new ColorMatrixColorFilter(matrix);
        paint.setColorFilter(filter);
        canvas.drawBitmap(src, 0, 0, paint);

        return dest;
    }

    private int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    private MultiBoxTracker tracker;
    private BlazeFace blazeFace;
    private FaceNet faceNet;
    private LibSVM svm;

    boolean scanData(int label, Bitmap bitmap) throws Exception {
        //ArrayList<float[]> list = new ArrayList<>();
        boolean isFaceEmpty = true;
        List<RectF> faces = blazeFace.detect(bitmap);
        isFaceEmpty = faces.isEmpty();
        Timber.d(TAG+"FACES_DETECTED - FACES_IS_EMPTY - "+isFaceEmpty);
        return isFaceEmpty;
    }

    private void disableCamera() {
        //running = false;
        if (cameraBridgeViewBase != null)
            cameraBridgeViewBase.disableView();
    }

    private void startTimer() {
        stoptimertask();
        timer = new Timer();
        runTimerTask();
        //schedule the timer, to wake up every 1 second
        timer.schedule(timerTask, 0, 1000);
    }

    private void runTimerTask() {
        timerTask = new TimerTask() {
            public void run() {
                counterTimer--;
                Timber.d(TAG+"RUN TIMER IS -- "+ counterTimer);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        textCountTime.setText(""+counterTimer);
                        if(counterTimer == 0)
                        {
                            textCountTime.setVisibility(View.GONE);
                            waitSeconds = false;
                            counterTimer = 7;
                            stoptimertask();
                        }
                    }
                });
            }
        };
    }

    private void stoptimertask() {
        //stop the timer, if it's not already null
        if (timer != null) {
            timer.cancel();
            timer = null;
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        matTmpProcessingFace = new Mat();
        matTmpProcessingRgbaFace = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        matTmpProcessingFace.release();
        matTmpProcessingRgbaFace.release();
    }

    @Override
    public void onPermissionsGranted(int requestCode, List<String> perms) {
        Log.d(TAG, "PERMISSION_GRANTED");
    }

    @Override
    public void onPermissionsDenied(int requestCode, List<String> perms) {
        Log.d(TAG, "PERMISSION_DENIED");
        finish();
    }

    @Override
    public void onActivityResume() {
        boolean permGranted = isPermissionGranted();
        Timber.d(TAG+"Permission ---- "+permGranted +" | "+ cameraBridgeViewBase);
        if (!permGranted) return;
        if (cameraBridgeViewBase != null)
            resumeOCV();
    }

    @Override
    public void onActivityPause() {
        disableCamera();
    }

    @Override
    public void onActivityStop() {

    }

    @Override
    public void onActivityDestroy() {
        disableCamera();
    }

    @Override
    public void onActivityKeyDown(int keyCode, KeyEvent event) {

    }

    @Override
    public void onActivityFinish() {

    }

    @Override
    public void onActivityRestart() {

    }

    @Override
    public void onActivitySaveInstanceState(Bundle outState) {

    }

    @Override
    public void onActivityRestoreInstanceState(Bundle savedInstanceState) {

    }

    @Override
    public void onActivityRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        //checkPermissions();

        Log.d(TAG, "PERMS_RESULT - "+requestCode +" | "+ PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6);
        if (requestCode != PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6) {
            Timber.d(TAG+"Got unexpected permission result: " + requestCode);
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            return;
        }

        if (grantResults.length != 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.d(TAG, "Camera permission granted - initialize the camera source");
            // we have permission, so create the camerasource
            loadCameraBridge();
            setInit();
            return;
        }

        Log.d(TAG, "Permission not granted: results len = " + grantResults.length +
                " Result code = " + (grantResults.length > 0 ? grantResults[0] : "(empty)"));

        finish();
    }

    @Override
    public void onActivityMResult(int requestCode, int resultCode, Intent data) {

    }

    @Override
    public void setAnimationOnOpenActivity(Integer firstAnim, Integer secondAnim) {

    }

    @Override
    public void setAnimationOnCloseActivity(Integer firstAnim, Integer secondAnim) {

    }

    @Override
    public View setViewTreeObserverActivity() {
        return null;
    }

    @Override
    public void getViewTreeObserverActivity() {

    }

    @Override
    public Intent setResultIntent() {
        return null;
    }

    @Override
    public String getTagDataIntentFromActivity() {
        return null;
    }

    @Override
    public void getMapDataIntentFromActivity(MapDataParcelable parcleable) {

    }

    @Override
    public MapDataParcelable setMapDataIntentToNextActivity(MapDataParcelable parcleable) {
        return null;
    }

    public class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        Recognition(
                final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }
}