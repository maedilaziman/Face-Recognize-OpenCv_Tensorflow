package br.com.helpdev.facedetect;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.maedi.soft.ino.base.utils.ScreenSize;
import com.maedi.soft.ino.recognize.face.R;
import com.maedi.soft.ino.recognize.face.Recognizer;
import com.maedi.soft.ino.recognize.face.env.FileUtils;
import com.maedi.soft.ino.recognize.face.ml.BlazeFace;
import com.maedi.soft.ino.recognize.face.ml.FaceNet;
import com.maedi.soft.ino.recognize.face.ml.LibSVM;
import com.maedi.soft.ino.recognize.face.recognizeupdate.SupportVectorMachine;
import com.maedi.soft.ino.recognize.face.recognizeupdate.rcgutils.FileHelper;
import com.maedi.soft.ino.recognize.face.recognizeupdate.rcgutils.MatName;
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
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;
import pub.devrel.easypermissions.EasyPermissions;
import timber.log.Timber;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, Runnable, EasyPermissions.PermissionCallbacks {

    private static final String TAG = "- OCVFaceDetect - ";
    private CameraBridgeViewBase cameraBridgeViewBase;
    private static final Scalar FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Log.d(TAG, "OpenCV loaded successfully");
                cameraBridgeViewBase.enableView();
            } else {
                super.onManagerConnected(status);
            }
        }
    };

    private volatile boolean running = false;
    private volatile int qtdFaces;
    private volatile Mat matTmpProcessingFace, matTmpProcessingRgbaFace;
    private float                  mRelativeFaceSize   = 0.2f;
    private int                    mAbsoluteFaceSize   = 0;

    private CascadeClassifier cascadeClassifier, cascadeClEyesRight, cascadeClEyesLeft;
    private File mCascadeFile, mCascadeFlEyesLeft, mCascadeFlEyesRight;
    private TextView infoFaces;
    private LinearLayout layoutRight, layoutAddname;
    private ImageView imageView1, imageView2, imageView3;
    private TextView textCountTime, textClearImg, textTrainImg;
    private EditText editName;
    private Button buttonAddname;
    private ProgressBar progressBar1;
    private FragmentActivity f;
    private boolean waitSeconds;
    private Timer timer;
    private TimerTask timerTask;
    private int counterTimer;
    private ArrayList<Bitmap> arrayListBitmap;
    private Recognizer recognizer;
    private FileHelper fh;
    private List<Mat> imagesRgbs;

    //private Map mparcel;
    private Intent args;
    private boolean isUnknowUser;
    private final String strUnknowUser = "zzzUnknow";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        f = this;
        arrayListBitmap = new ArrayList();
        infoFaces = findViewById(R.id.tv);
        layoutRight = findViewById(R.id.layout_right);
        layoutAddname = findViewById(R.id.layout_addname);
        cameraBridgeViewBase = findViewById(R.id.main_surface);
        imageView1 = findViewById(R.id.image1);
        imageView2 = findViewById(R.id.image2);
        imageView3 = findViewById(R.id.image3);
        textCountTime = findViewById(R.id.count_time);
        textClearImg = findViewById(R.id.clear_imgdata);
        textTrainImg = findViewById(R.id.train_imgdata);
        editName = findViewById(R.id.edit_name);
        buttonAddname = findViewById(R.id.buttonAddname);
        progressBar1 = findViewById(R.id.progressBar1);

        waitSeconds = false;
        counterTimer = 7;
        fh = new FileHelper();
        imagesRgbs = new ArrayList<>();
        args = getIntent();

        Timber.d(TAG+"ARGS -> "+args);
        if(null != args)
        {
            isUnknowUser = args.getBooleanExtra("unknow_user", false);
            Timber.d(TAG+"ARGS_isUnknowUser -> "+isUnknowUser);
            if(isUnknowUser)
            {
                imageView2.setVisibility(View.GONE);
                imageView3.setVisibility(View.GONE);
                editName.setText(strUnknowUser);
                editName.setEnabled(false);
            }
        }

        RelativeLayout.LayoutParams layoutParams = new RelativeLayout.LayoutParams(RelativeLayout.LayoutParams.MATCH_PARENT, RelativeLayout.LayoutParams.WRAP_CONTENT);
        layoutParams.width = ScreenSize.instance(f).getWidth()/2;
        cameraBridgeViewBase.setLayoutParams(layoutParams);

        LinearLayout.LayoutParams layoutParams2 = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        layoutParams2.width = ScreenSize.instance(f).getWidth()/2;
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

    private void setInit()
    {
        init();
        loadHaarCascadeFile();
        resumeOCV();
        textClearImg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                clearDataPerson();
            }
        });

        textTrainImg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //imgTrain(1);
            }
        });

        buttonAddname.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String text = editName.getText().toString();
                if(text.length() > 0)
                {
                    progressBar1.setVisibility(View.VISIBLE);
                    enabledButton(false);

                    //add name to file
                    //int idx = recognizer.addPerson(text);
                    //Log.d(TAG, "INDEX_PERSON - "+idx);
                    //new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
                    //    @Override
                    //    public void run() {
                    //        imgTrain(idx);
                    //    }
                    //}, 600);

                    int j=1;
                    String wholeFolderPath = fh.TRAINING_PATH + text;
                    //new File(wholeFolderPath).mkdirs();
                    File dir = new File(wholeFolderPath);
                    if (!dir.isDirectory()) {
                        dir.mkdirs();
                    }
                    else
                    {
                        Toast.makeText(f, "The Directory is exists!", Toast.LENGTH_SHORT).show();
                        return;
                    }

                    for(Mat img : imagesRgbs)
                    {
                        MatName m = new MatName(text + "_"+j, img);
                        //String wholeFolderPath = fh.TRAINING_PATH + text;
                        //new File(wholeFolderPath).mkdirs();
                        fh.saveMatToImage(m, wholeFolderPath + "/");
                        j++;
                    }

                    new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
                        @Override
                        public void run() {
                            FileHelper fileHelper = new FileHelper();
                            fileHelper.createDataFolderIfNotExsiting();
                            final File[] persons = fileHelper.getTrainingList();
                            SupportVectorMachine svm = new SupportVectorMachine(f, SupportVectorMachine.TRAINING);
                            int x = 0;
                            Timber.d(TAG+"PERSON_LENGTH -> "+persons.length);
                            if (persons.length > 0) {
                                for (File person : persons) {
                                    Timber.d(TAG+"PERSON_FILE IS DIR ? -> "+person.isDirectory());
                                    if (person.isDirectory()) {
                                        File[] files = person.listFiles();
                                        Timber.d(TAG+"PERSON_FILE LIST -> "+files.length);
                                        int counter = 1;
                                        for (File file : files) {
                                            Timber.d(TAG+"PERSON_FILE DATA -> "+file.getAbsolutePath());
                                            boolean isAnImg = FileHelper.isFileAnImage(file);
                                            Timber.d(TAG+"PERSON_FILE isAnImg -> "+isAnImg);
                                            if (isAnImg){
                                                Mat imgRgb = Imgcodecs.imread(file.getAbsolutePath());
                                                Imgproc.cvtColor(imgRgb, imgRgb, Imgproc.COLOR_BGRA2RGBA);
                                                Mat processedImage = new Mat();
                                                imgRgb.copyTo(processedImage);

                                                // The last token is the name --> Folder name = Person name
                                                String[] tokens = file.getParent().split("/");
                                                final String name = tokens[tokens.length - 1];
                                                Timber.d(TAG+"PERSON_FILE Label Name -> "+name);

                                                MatName m = new MatName("processedImage", processedImage);
                                                Timber.d(TAG+"PERSON_FILE MatName -> "+m.getName());
                                                String strFullPath = fileHelper.saveMatToImage(m, FileHelper.DATA_PATH);
                                                Timber.d(TAG+"PERSON_FILE FULL_PATH -> "+strFullPath);
                                                svm.addImage(processedImage, name, false);
                                                Timber.d(TAG+"PERSON_FILE Success Added To SVM ...");
                                            }
                                        }
                                    }
                                }
                                x++;
                                if(x == persons.length)
                                {
                                    Timber.d(TAG+"Start Train...");
                                    //Toast.makeText(f, "Start Train...", Toast.LENGTH_SHORT).show();
                                //    clearDataPerson();
                                //    progressBar1.setVisibility(View.GONE);
                                //    enabledButton(true);
                                }
                            }
                            if(svm.train())
                            //if(svm.trainProbability("-t 0"))
                            {
                                Toast.makeText(f, "Train Successfull...", Toast.LENGTH_LONG).show();
                                clearDataPerson();
                                progressBar1.setVisibility(View.GONE);
                                enabledButton(true);
                            }
                        }
                    }, 1000);
                }
                else {
                    Toast.makeText(f, "Name cannot be empty!", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void disActivateEdtName()
    {
        if(isUnknowUser)
        {
            editName.setText(strUnknowUser);
            editName.setEnabled(false);
        }
    }

    private void clearDataPerson()
    {
        disActivateEdtName();
        imageView1.setImageDrawable(f.getDrawable(R.drawable.no_media));
        imageView2.setImageDrawable(f.getDrawable(R.drawable.no_media));
        imageView3.setImageDrawable(f.getDrawable(R.drawable.no_media));
        imageView1.setTag(null);
        imageView2.setTag(null);
        imageView3.setTag(null);
        layoutAddname.setVisibility(View.GONE);
        editName.setText("");
        arrayListBitmap.clear();
        imagesRgbs.clear();
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
            Log.e(TAG, "Exception initializing classifier! -> "+e.getMessage());
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

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
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

    private void loadCameraBridge() {
        cameraBridgeViewBase.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
    }

    private void loadHaarCascadeFile() {
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
            /*
            // ------------------------- load right eye classificator-----------------------//
            File cascadeDirER = getDir("haarcascade_eye_right", Context.MODE_PRIVATE);
            mCascadeFlEyesRight = new File(cascadeDirER, "haarcascade_eye_right.xml");
            if (!mCascadeFlEyesRight.exists()) {
                InputStream iser = getResources().openRawResource(R.raw.haarcascade_righteye_2splits);
                FileOutputStream oser = new FileOutputStream(mCascadeFlEyesRight);

                byte[] bufferER = new byte[4096];
                int bytesReadER;
                while ((bytesReadER = iser.read(bufferER)) != -1) {
                    oser.write(bufferER, 0, bytesReadER);
                }
                iser.close();
                oser.close();
            }
            // ------------------------- load left eye classificator-----------------------//
            File cascadeDirEL = getDir("haarcascade_eye_left", Context.MODE_PRIVATE);
            mCascadeFlEyesLeft = new File(cascadeDirEL, "haarcascade_eye_left.xml");
            if (!mCascadeFlEyesLeft.exists()) {
                InputStream isel = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
                FileOutputStream osel = new FileOutputStream(mCascadeFlEyesLeft);

                byte[] bufferEL = new byte[4096];
                int bytesReadEL;
                while ((bytesReadEL = isel.read(bufferEL)) != -1) {
                    osel.write(bufferEL, 0, bytesReadEL);
                }
                isel.close();
                osel.close();
            }
            */

        } catch (Throwable throwable) {
            throw new RuntimeException("Failed to load Haar Cascade file");
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        disableCamera();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!isPermissionGranted()) return;
        if (cameraBridgeViewBase != null)
            resumeOCV();
    }

    private void resumeOCV() {
        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        }
        cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
        cascadeClassifier.load(mCascadeFile.getAbsolutePath());

        //cascadeClEyesRight = new CascadeClassifier(mCascadeFlEyesRight.getAbsolutePath());
        //cascadeClEyesLeft = new CascadeClassifier(mCascadeFlEyesLeft.getAbsolutePath());
        //cascadeClEyesRight.load(mCascadeFlEyesRight.getAbsolutePath());
        //cascadeClEyesLeft.load(mCascadeFlEyesLeft.getAbsolutePath());
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
                Bitmap grayScale = getGrayscale(bitmap);
                Utils.matToBitmap(nwmath, grayScale);

                try {
                    boolean isEmptyImage = scanData(1, grayScale);
                    Log.d(TAG, "isEmptyImage:"+isEmptyImage);
                    if(!isEmptyImage)
                    {
                        Log.d(TAG, "imageView -> getTag():"+imageView1.getTag() +" | "+ imageView1.getTag() +" | "+ imageView1.getTag());

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                if(!isUnknowUser) {
                                    if (null == imageView1.getTag()) {
                                        imageView1.setImageBitmap(grayScale);
                                        setBW(imageView1);
                                        imageView1.setTag("1");
                                        arrayListBitmap.add(grayScale);
                                        Mat mat = new Mat();
                                        Utils.bitmapToMat(grayScale, mat);
                                        imagesRgbs.add(mat);
                                    } else if (null == imageView2.getTag()) {
                                        imageView2.setImageBitmap(grayScale);
                                        setBW(imageView2);
                                        imageView2.setTag("1");
                                        arrayListBitmap.add(grayScale);
                                        Mat mat = new Mat();
                                        Utils.bitmapToMat(grayScale, mat);
                                        imagesRgbs.add(mat);
                                    } else if (null == imageView3.getTag()) {
                                        imageView3.setImageBitmap(grayScale);
                                        setBW(imageView3);
                                        imageView3.setTag("1");
                                        arrayListBitmap.add(grayScale);
                                        Mat mat = new Mat();
                                        Utils.bitmapToMat(grayScale, mat);
                                        imagesRgbs.add(mat);

                                        layoutAddname.setVisibility(View.VISIBLE);
                                    }
                                }
                                else
                                {
                                    imageView1.setImageBitmap(grayScale);
                                    setBW(imageView1);
                                    arrayListBitmap.add(grayScale);
                                    Mat mat = new Mat();
                                    Utils.bitmapToMat(grayScale, mat);
                                    imagesRgbs.add(mat);

                                    if(imagesRgbs.size() >= 10)
                                        layoutAddname.setVisibility(View.VISIBLE);

                                }

                                waitSeconds = true;
                                textCountTime.setVisibility(View.VISIBLE);
                                startTimer();
                            }
                        });
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    Timber.d(TAG+"Exception:"+e.getMessage());
                }
            }

            //Log.d(TAG, "FACE_ARRAY_COUNT:"+facesArray.length);
            for (int i = 0; i < facesArray.length; i++) {
                Imgproc.rectangle(matTmpProcessingRgbaFace, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
            }
        }

        return matTmpProcessingRgbaFace;
    }

    public Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
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

    private void setBW(ImageView iv){

        float[] matrix = new float[]{
                0.3f, 0.59f, 0.11f, 0, 0,
                0.3f, 0.59f, 0.11f, 0, 0,
                0.3f, 0.59f, 0.11f, 0, 0,
                0, 0, 0, 1, 0,};

        ColorFilter colorFilter = new ColorMatrixColorFilter(matrix);
        iv.setColorFilter(colorFilter);
    }

    /*
    boolean updateData(int label, ContentResolver contentResolver, ArrayList<Uri> uris) throws Exception {
        synchronized (this) {
            ArrayList<float[]> list = new ArrayList<>();
            boolean isFaceEmpty = true;
            for (Uri uri : uris) {
                Timber.d("FACES_DETECTED - URI - "+uri.toString());
                Bitmap bitmap = getBitmapFromUri(contentResolver, uri);
                List<RectF> faces = blazeFace.detect(bitmap);
                isFaceEmpty = faces.isEmpty();
                Timber.d("FACES_DETECTED - FACES_IS_EMPTY - "+isFaceEmpty);
                Rect rect = new Rect();
                if (!isFaceEmpty) {
                    faces.get(0).round(rect);
                    float[] emb_array = new float[FaceNet.EMBEDDING_SIZE];
                    faceNet.getEmbeddings(bitmap, rect).get(emb_array);
                    list.add(emb_array);
                }
                else
                {
                    break;
                }
            }

            if (!isFaceEmpty)svm.train(label, list);

            return isFaceEmpty;
        }
    }
    */

    private void imgTrain(int label)
    {
        ArrayList<float[]> list = new ArrayList<>();
        int x = 0;
        for (Bitmap bitmap : arrayListBitmap) {
            List<RectF> faces = blazeFace.detect(bitmap);
            Rect rect = new Rect();
            faces.get(0).round(rect);
            float[] emb_array = new float[FaceNet.EMBEDDING_SIZE];
            faceNet.getEmbeddings(bitmap, rect).get(emb_array);
            list.add(emb_array);
            x++;
            if(x == arrayListBitmap.size())
            {
                Toast.makeText(f, "Train Successfull...", Toast.LENGTH_LONG).show();
                clearDataPerson();
                progressBar1.setVisibility(View.GONE);
                enabledButton(true);
            }
        }

        svm.train(label, list);
    }

    private void enabledButton(boolean b)
    {
        buttonAddname.setEnabled(b);
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


    public void startFaceDetect() {
        if (running) return;
        new Thread(this).start();
    }

    @Override
    public void run() {
        running = true;
        while (running) {
            try {
                if (matTmpProcessingFace != null) {
                    MatOfRect matOfRect = new MatOfRect();
                    cascadeClassifier.detectMultiScale(matTmpProcessingFace, matOfRect);
                    //org.opencv.core.Rect[] nrect = matOfRect.toArray();
                    //String strRect = nrect.toString();
                    //Log.d("RECT", strRect);
                    Log.d(TAG, "matTmpProcessingRgbaFace:"+matTmpProcessingRgbaFace);
                    int newQtdFaces = matOfRect.toList().size();
                    Log.d(TAG, "FACE_COUNT:"+newQtdFaces +" | "+ qtdFaces);
                    if(newQtdFaces == 1)
                    {
                        if(null != matTmpProcessingRgbaFace) {
                            org.opencv.core.Rect[] facesArray = matOfRect.toArray();
                            Log.d(TAG, "FACE_ARRAY_COUNT:"+facesArray.length);
                            for (int i = 0; i < facesArray.length; i++) {
                                Imgproc.rectangle(matTmpProcessingRgbaFace, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
                            }
                            //Imgproc.rectangle(matTmpProcessingRgbaFace, new Point(10, 100), new Point(20, 100),new Scalar(0, 255, 0));
                            //Imgproc.putText(matTmpProcessingRgbaFace, "====", new Point(10,100), 3, 1, new Scalar(255, 0, 0, 255), 2);
                        }
                    }
                    if (qtdFaces != newQtdFaces) {
                        qtdFaces = newQtdFaces;
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                infoFaces.setText(String.format(getString(R.string.faces_detects), qtdFaces));
                            }
                        });
                    }
                    Thread.sleep(2000);//if you want an interval
                    matTmpProcessingFace = null;
                    //matTmpProcessingRgbaFace = null;
                }
                Thread.sleep(200);
            } catch (Throwable t) {
                try {
                    Thread.sleep(10_000);
                } catch (Throwable tt) {
                }
            }
        }
    }

    private MultiBoxTracker tracker;
    private BlazeFace blazeFace;
    private FaceNet faceNet;
    private LibSVM svm;

    boolean scanData(int label, Bitmap bitmap) throws Exception {
        /*
        File dir = new File(FileUtils.ROOT);
        if (!dir.isDirectory()) {
            if (dir.exists()) dir.delete();
            dir.mkdirs();

            AssetManager mgr = getAssets();
            FileUtils.copyAsset(mgr, FileUtils.DATA_FILE);
            FileUtils.copyAsset(mgr, FileUtils.MODEL_FILE);
            FileUtils.copyAsset(mgr, FileUtils.LABEL_FILE);
        }
        blazeFace = BlazeFace.create(getAssets());
        //faceNet = FaceNet.create(getAssets());
        //svm = LibSVM.getInstance();
        */

        //ArrayList<float[]> list = new ArrayList<>();
        boolean isFaceEmpty = true;
        List<RectF> faces = blazeFace.detect(bitmap);
        isFaceEmpty = faces.isEmpty();
        Timber.d(TAG+"FACES_DETECTED - FACES_IS_EMPTY - "+isFaceEmpty);
        //Rect rect = new Rect();
        //if (!isFaceEmpty) {
        //    Toast.makeText(f, "FACE_SUCCESS_DETECTED", Toast.LENGTH_SHORT).show();
        //    faces.get(0).round(rect);
        //    float[] emb_array = new float[FaceNet.EMBEDDING_SIZE];
        //    faceNet.getEmbeddings(bitmap, rect).get(emb_array);
        //    list.add(emb_array);
        //}
        //if (!isFaceEmpty)svm.train(label, list);

        return isFaceEmpty;
    }

    public void onDestroy() {
        super.onDestroy();
        disableCamera();
    }

    private void disableCamera() {
        running = false;
        if (cameraBridgeViewBase != null)
            cameraBridgeViewBase.disableView();
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
}