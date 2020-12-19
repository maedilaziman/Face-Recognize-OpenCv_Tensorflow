package com.maedi.soft.ino.recognize.face;

import android.Manifest;
import android.app.Activity;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.Toast;

import com.google.android.gms.common.images.Size;
import com.google.gson.Gson;
import com.maedi.soft.ino.base.BaseFragment;
import com.maedi.soft.ino.base.BuildActivity;
import com.maedi.soft.ino.base.func_interface.ActivityListener;
import com.maedi.soft.ino.base.store.MapDataParcelable;
import com.maedi.soft.ino.recognize.face.adapter.RecyclerPreviewPhotoCamera;
import com.maedi.soft.ino.recognize.face.env.FileUtils;
import com.maedi.soft.ino.recognize.face.env.ImageUtils;
import com.maedi.soft.ino.recognize.face.ml.BlazeFace;
import com.maedi.soft.ino.recognize.face.ml.FaceNet;
import com.maedi.soft.ino.recognize.face.ml.LibSVM;
import com.maedi.soft.ino.recognize.face.model.ListObject;
import com.maedi.soft.ino.recognize.face.tracking.MultiBoxTracker;
import com.maedi.soft.ino.recognize.face.utils.DataStatic;
import com.maedi.soft.ino.recognize.face.utils.FuncHelper;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;
import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import butterknife.BindView;
import butterknife.ButterKnife;
import timber.log.Timber;

import static android.app.Activity.RESULT_CANCELED;
import static android.app.Activity.RESULT_OK;

public class OpenFaceCamera extends BuildActivity<View> implements ActivityListener<Integer>, RecyclerPreviewPhotoCamera.CommPreviewPhoto, Camera.PictureCallback, SurfaceHolder.Callback {

    private final String TAG = this.getClass().getName()+"- OPEN_FACE_CAMERAS - ";

    @BindView(R.id.preview_view)
    SurfaceView mCameraPreview;

    @BindView(R.id.capture_image_button)
    Button btnCaptureImage;

    @BindView(R.id.listView)
    RecyclerView listView;

    @BindView(R.id.progressBar)
    ProgressBar progressBar;

    private FragmentActivity f;

    private RecyclerPreviewPhotoCamera previewCameraAdapter;

    private ListObject listObject;

    private Camera mCamera;

    private byte[] mCameraData;

    private boolean mIsCapturing;

    private Map<String, String> mPathResultImage;

    private int mkey;

    private int currentCameraId = 0;

    private Map tempPhoto;

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
        return R.layout.activity_open_face_camera;
    }

    @Override
    public ActivityListener createListenerForActivity() {
        return this;
    }

    @Override
    public void onCreateActivity(Bundle savedInstanceState) {
        f = this;
        ButterKnife.bind(this);

        LinearLayoutManager layoutManager = new LinearLayoutManager(f, LinearLayoutManager.HORIZONTAL, false);
        listView.setLayoutManager(layoutManager);

        listObject = new ListObject();
        previewCameraAdapter = new RecyclerPreviewPhotoCamera(f, f, R.layout.list_preview_photo_camera, listObject, this, "");
        listView.setAdapter(previewCameraAdapter);
    }

    @Override
    public void onBuildActivityCreated() {
        init();
    }

    private void init()
    {

        mPathResultImage = new HashMap<String, String>();
        mkey = 0;
        mIsCapturing = true;
        tempPhoto = new HashMap<>();
        SurfaceHolder surfaceHolder = mCameraPreview.getHolder();
        surfaceHolder.addCallback(this);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

        //int rc = ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA);
        //if (rc == PackageManager.PERMISSION_GRANTED) {
        //    setCameras();
        //}
        //else {
        //    requestCameraPermission();
        //}
        mFrameProcessor = new FrameProcessingRunnable();
    }

    private void requestCameraPermission() {
        Timber.d(TAG+"Camera permission is not granted. Requesting permission");
        if (FuncHelper.hasAPI_LEVEL24_ANDROID_7_Above())
            FuncHelper.CameraPermission_API_LEVEL24_ANDROID_7_Above(f);
        else FuncHelper.CameraPermission(f);
    }

    private void start(SurfaceHolder surfaceHolder) throws IOException {
        setCameras();
        Timber.d(TAG+"START_PREVIEW start() camera -> "+mCamera);
        if(null != mCamera) {
            Timber.d(TAG+"START_PREVIEW");
            synchronized (mCameraLock) {
                // SurfaceTexture was introduced in Honeycomb (11), so if we are running and
                // old version of Android. fall back to use SurfaceView.
                if(null == surfaceHolder) {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB) {
                        mDummySurfaceTexture = new SurfaceTexture(DUMMY_TEXTURE_NAME);
                        mCamera.setPreviewTexture(mDummySurfaceTexture);
                    } else {
                        mDummySurfaceView = new SurfaceView(f);
                        mCamera.setPreviewDisplay(mDummySurfaceView.getHolder());
                    }
                    mCamera.startPreview();
                }
                else
                {
                    mCamera.setPreviewDisplay(surfaceHolder);
                    mCamera.startPreview();
                }
                setPrevCallBack();

                mProcessingThread = new Thread(mFrameProcessor);
                mFrameProcessor.setActive(true);
                mProcessingThread.start();

            }
        }
    }

    private void stop() {
        synchronized (mCameraLock) {
            mFrameProcessor.setActive(false);
            if (mProcessingThread != null) {
                try {
                    // Wait for the thread to complete to ensure that we can't have multiple threads
                    // executing at the same time (i.e., which would happen if we called start too
                    // quickly after stop).
                    mProcessingThread.join();
                } catch (InterruptedException e) {
                    Log.d(TAG, "Frame processing thread interrupted on release.");
                }
                mProcessingThread = null;
            }

            if (mCamera != null) {
                mCamera.stopPreview();
                mCamera.setPreviewCallbackWithBuffer(null);
                try {
                    // We want to be compatible back to Gingerbread, but SurfaceTexture
                    // wasn't introduced until Honeycomb.  Since the interface cannot use a SurfaceTexture, if the
                    // developer wants to display a preview we must use a SurfaceHolder.  If the developer doesn't
                    // want to display a preview we use a SurfaceTexture if we are running at least Honeycomb.

                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB) {
                        mCamera.setPreviewTexture(null);

                    } else {
                        mCamera.setPreviewDisplay(null);
                    }
                } catch (Exception e) {
                    Log.e(TAG, "Failed to clear camera preview: " + e);
                }
                mCamera.release();
                mCamera = null;
            }
        }
    }

    private void setCameras()
    {
        int rc = ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA);
        if (rc != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission();
            return;
        }

        if (mCamera != null)
        {
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;

        }

        //if (currentCameraId == Camera.CameraInfo.CAMERA_FACING_BACK) {
        //    currentCameraId = Camera.CameraInfo.CAMERA_FACING_FRONT;
        //} else {
        //    currentCameraId = Camera.CameraInfo.CAMERA_FACING_BACK;
        //}
        currentCameraId = Camera.CameraInfo.CAMERA_FACING_BACK;
        mCamera = Camera.open(currentCameraId);

        if (mCamera != null) {
            try {
                Camera.Parameters parameters = mCamera.getParameters();

                parameters.set("orientation", "portrait");
                parameters.set("rotation", 90);
                //List<Camera.Size> sizeList = parameters.getSupportedPictureSizes();
                //int chosenSize = FuncHelper.getPictureSizeIndexForHeight(sizeList, 800);
                //parameters.setPictureSize(sizeList.get(chosenSize).width, sizeList.get(chosenSize).height);

                SizePair sizePair = selectSizePair(mCamera, mRequestedPreviewWidth, mRequestedPreviewHeight);
                if (sizePair == null) {
                    throw new RuntimeException("Could not find suitable preview size.");
                }
                Size pictureSize = sizePair.pictureSize();
                mPreviewSize = sizePair.previewSize();
                mRequestedFps = 15.0f;
                int[] previewFpsRange = selectPreviewFpsRange(mCamera, mRequestedFps);
                if (previewFpsRange == null) throw new RuntimeException("Could not find suitable preview frames per second range.");
                if (pictureSize != null) parameters.setPictureSize(pictureSize.getWidth(), pictureSize.getHeight());

                parameters.setPreviewSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
                parameters.setPreviewFpsRange(
                        previewFpsRange[Camera.Parameters.PREVIEW_FPS_MIN_INDEX],
                        previewFpsRange[Camera.Parameters.PREVIEW_FPS_MAX_INDEX]);
                parameters.setPreviewFormat(ImageFormat.NV21);

                //mCamera.setDisplayOrientation(90);
                setCameraDisplayOrientation(f, currentCameraId, mCamera, parameters);
                //mCamera.setPreviewDisplay(mCameraPreview.getHolder());
                mCamera.setParameters(parameters);

            }
            catch (Exception e) {
                Timber.d(TAG+"Cannot open cameras! -> "+e.getMessage());
                Toast.makeText(f, "Cannot open cameras! -> "+e.getLocalizedMessage(), Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void setPrevCallBack()
    {
        // Four frame buffers are needed for working with the camera:
        //
        //   one for the frame that is currently being executed upon in doing detection
        //   one for the next pending frame to process immediately upon completing detection
        //   two for the frames that the camera uses to populate future preview images
        mCamera.setPreviewCallbackWithBuffer(new CameraPreviewCallback());
        mCamera.addCallbackBuffer(createPreviewBuffer(mPreviewSize));
        //mCamera.addCallbackBuffer(createPreviewBuffer(mPreviewSize));
        //mCamera.addCallbackBuffer(createPreviewBuffer(mPreviewSize));
        //mCamera.addCallbackBuffer(createPreviewBuffer(mPreviewSize));
    }

    /**
     * The dummy surface texture must be assigned a chosen name.  Since we never use an OpenGL
     * context, we can choose any ID we want here.
     */
    private static final int DUMMY_TEXTURE_NAME = 100;
    private static final float ASPECT_RATIO_TOLERANCE = 0.01f;
    private Size mPreviewSize;
    private float mRequestedFps = 30.0f;
    private int mRequestedPreviewWidth = 1024;
    private int mRequestedPreviewHeight = 1768;
    private SurfaceView mDummySurfaceView;
    private SurfaceTexture mDummySurfaceTexture;
    private Thread mProcessingThread;
    private FrameProcessingRunnable mFrameProcessor;
    private final Object mCameraLock = new Object();

    private byte[] createPreviewBuffer(Size previewSize) {
        int bitsPerPixel = ImageFormat.getBitsPerPixel(ImageFormat.NV21);
        long sizeInBits = previewSize.getHeight() * previewSize.getWidth() * bitsPerPixel;
        int bufferSize = (int) Math.ceil(sizeInBits / 8.0d) + 1;

        //
        // NOTICE: This code only works when using play services v. 8.1 or higher.
        //

        // Creating the byte array this way and wrapping it, as opposed to using .allocate(),
        // should guarantee that there will be an array to work with.
        byte[] byteArray = new byte[bufferSize];
        ByteBuffer buffer = ByteBuffer.wrap(byteArray);
        if (!buffer.hasArray() || (buffer.array() != byteArray)) {
            // I don't think that this will ever happen.  But if it does, then we wouldn't be
            // passing the preview content to the underlying detector later.
            throw new IllegalStateException("Failed to create valid buffer for camera source.");
        }
        return byteArray;
    }

    private ByteBuffer getByteBuffer(byte[] byteArray)
    {
        ByteBuffer buffer = ByteBuffer.wrap(byteArray);
        return buffer;
    }

    private static class SizePair {
        private Size mPreview;
        private Size mPicture;

        public SizePair(Camera.Size previewSize,
                        Camera.Size pictureSize) {
            mPreview = new Size(previewSize.width, previewSize.height);
            if (pictureSize != null) {
                mPicture = new Size(pictureSize.width, pictureSize.height);
            }
        }

        public Size previewSize() {
            return mPreview;
        }

        @SuppressWarnings("unused")
        public Size pictureSize() {
            return mPicture;
        }
    }

    private static SizePair selectSizePair(Camera camera, int desiredWidth, int desiredHeight) {
        List<SizePair> validPreviewSizes = generateValidPreviewSizeList(camera);

        // The method for selecting the best size is to minimize the sum of the differences between
        // the desired values and the actual values for width and height.  This is certainly not the
        // only way to select the best size, but it provides a decent tradeoff between using the
        // closest aspect ratio vs. using the closest pixel area.
        SizePair selectedPair = null;
        int minDiff = Integer.MAX_VALUE;
        for (SizePair sizePair : validPreviewSizes) {
            Size size = sizePair.previewSize();
            int diff = Math.abs(size.getWidth() - desiredWidth) +
                    Math.abs(size.getHeight() - desiredHeight);
            if (diff < minDiff) {
                selectedPair = sizePair;
                minDiff = diff;
            }
        }

        return selectedPair;
    }

    private static List<SizePair> generateValidPreviewSizeList(Camera camera) {
        Camera.Parameters parameters = camera.getParameters();
        List<Camera.Size> supportedPreviewSizes =
                parameters.getSupportedPreviewSizes();
        List<Camera.Size> supportedPictureSizes =
                parameters.getSupportedPictureSizes();
        List<SizePair> validPreviewSizes = new ArrayList<>();
        for (Camera.Size previewSize : supportedPreviewSizes) {
            float previewAspectRatio = (float) previewSize.width / (float) previewSize.height;

            // By looping through the picture sizes in order, we favor the higher resolutions.
            // We choose the highest resolution in order to support taking the full resolution
            // picture later.
            for (Camera.Size pictureSize : supportedPictureSizes) {
                float pictureAspectRatio = (float) pictureSize.width / (float) pictureSize.height;
                if (Math.abs(previewAspectRatio - pictureAspectRatio) < ASPECT_RATIO_TOLERANCE) {
                    validPreviewSizes.add(new SizePair(previewSize, pictureSize));
                    break;
                }
            }
        }

        // If there are no picture sizes with the same aspect ratio as any preview sizes, allow all
        // of the preview sizes and hope that the camera can handle it.  Probably unlikely, but we
        // still account for it.
        if (validPreviewSizes.size() == 0) {
            //Timber.d(TAG+"No preview sizes have a corresponding same-aspect-ratio picture size");
            for (Camera.Size previewSize : supportedPreviewSizes) {
                // The null picture size will let us know that we shouldn't set a picture size.
                validPreviewSizes.add(new SizePair(previewSize, null));
            }
        }

        return validPreviewSizes;
    }

    private int[] selectPreviewFpsRange(Camera camera, float desiredPreviewFps) {
        // The camera API uses integers scaled by a factor of 1000 instead of floating-point frame
        // rates.
        int desiredPreviewFpsScaled = (int) (desiredPreviewFps * 1000.0f);

        // The method for selecting the best range is to minimize the sum of the differences between
        // the desired value and the upper and lower bounds of the range.  This may select a range
        // that the desired value is outside of, but this is often preferred.  For example, if the
        // desired frame rate is 29.97, the range (30, 30) is probably more desirable than the
        // range (15, 30).
        int[] selectedFpsRange = null;
        int minDiff = Integer.MAX_VALUE;
        List<int[]> previewFpsRangeList = camera.getParameters().getSupportedPreviewFpsRange();
        for (int[] range : previewFpsRangeList) {
            int deltaMin = desiredPreviewFpsScaled - range[Camera.Parameters.PREVIEW_FPS_MIN_INDEX];
            int deltaMax = desiredPreviewFpsScaled - range[Camera.Parameters.PREVIEW_FPS_MAX_INDEX];
            int diff = Math.abs(deltaMin) + Math.abs(deltaMax);
            if (diff < minDiff) {
                selectedFpsRange = range;
                minDiff = diff;
            }
        }
        return selectedFpsRange;
    }

    private class CameraPreviewCallback implements Camera.PreviewCallback {
        @Override
        public void onPreviewFrame(byte[] data, Camera camera) {
            Timber.d(TAG+"---onPreviewFrame_setNextFrame---");
            mFrameProcessor.setNextFrame(data, camera);
        }
    }

    private class FrameProcessingRunnable implements Runnable {
        private long mStartTimeMillis = SystemClock.elapsedRealtime();

        // This lock guards all of the member variables below.
        private final Object mLock = new Object();
        private boolean mActive = true;

        // These pending variables hold the state associated with the new frame awaiting processing.
        private long mPendingTimeMillis;
        private int mPendingFrameId = 0;
        private ByteBuffer mPendingFrameData;

        FrameProcessingRunnable() {
        }

        void setActive(boolean active) {
            synchronized (mLock) {
                mActive = active;
                mLock.notifyAll();
            }
        }

        /**
         * Sets the frame data received from the camera.  This adds the previous unused frame buffer
         * (if present) back to the camera, and keeps a pending reference to the frame data for
         * future use.
         */
        void setNextFrame(byte[] data, Camera camera) {
            Timber.d(TAG+"---Run_setNextFrame---");
            synchronized (mLock) {
                Timber.d(TAG+"---Run_setNextFrame--- mPendingFrameData= "+mPendingFrameData);
                if (mPendingFrameData != null) {
                    camera.addCallbackBuffer(mPendingFrameData.array());
                    mPendingFrameData = null;
                }

                // Timestamp and frame ID are maintained here, which will give downstream code some
                // idea of the timing of frames received and when frames were dropped along the way.
                mPendingTimeMillis = SystemClock.elapsedRealtime() - mStartTimeMillis;
                mPendingFrameData = getByteBuffer(createPreviewBuffer(mPreviewSize));
                //Camera.Parameters parameters = camera.getParameters();
                //int width = parameters.getPreviewSize().width;
                //int height = parameters.getPreviewSize().height;
                //Timber.d(TAG+"CAMERA_PREV_WIDTH - "+width +" <> "+ height);
                //YuvImage yuv = new YuvImage(data, parameters.getPreviewFormat(), width, height, null);
                //ByteArrayOutputStream out = new ByteArrayOutputStream();
                //yuv.compressToJpeg(new Rect(0, 0, width, height), 50, out);
                //byte[] bytes = out.toByteArray();
                //Bitmap bm = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
                //try {
                //    updateData(1, bm);
                //} catch (Exception e) {
                //    e.printStackTrace();
                //}
                /*
                if(null != bm) {
                    if ((bm.getWidth() > 200) && (bm.getHeight() > 200)) {
                        float darkThreshold = bm.getWidth() * bm.getHeight() * 0.35f;
                        int darkPixels = 0;
                        int[] pixels = new int[bm.getWidth() * bm.getHeight()];
                        bm.getPixels(pixels, 0, bm.getWidth(), 0, 0, bm.getWidth(), bm.getHeight());
                        boolean isbreak = false;
                        for (int j=0; j<pixels.length; j++) {
                            int color = pixels[j];
                            int r = Color.red(color);
                            int g = Color.green(color);
                            int b = Color.blue(color);
                            //Log.d(TAG, "DATA_PREVIEW_CAMERA - IS DARK RGB -> "+r +" | "+ g +" | "+ b);
                            double luminance = (((0.299 * r) + 0.0f) + ((0.587 * g) + 0.0f) + ((0.114 * b) + 0.0f));
                            if (luminance < 16) {
                                darkPixels++;
                            }
                        }
                        Log.d(TAG, "DATA_PREVIEW_CAMERA - #### -> " + darkPixels + " = " + darkThreshold);
                        if (darkPixels >= darkThreshold) {
                            if(null != listener)listener.backgroundLight(0);
                        } else {
                            if(null != listener)listener.backgroundLight(1);
                        }
                    }
                }
                */
                // Notify the processor thread if it is waiting on the next frame (see below).
                mLock.notifyAll();
            }
        }

        @Override
        public void run() {
            ByteBuffer data;
            Timber.d(TAG+"---RUN_THREAD---");
            while (true) {
                synchronized (mLock) {
                    Timber.d(TAG+"---RUN_THREAD--- mActive= "+mActive +" | "+ mPendingFrameData);
                    while (mActive && (mPendingFrameData == null)) {
                        try {
                            // Wait for the next frame to be received from the camera, since we
                            // don't have it yet.
                            mLock.wait();
                        } catch (InterruptedException e) {
                            Timber.d(TAG+"Frame processing loop terminated -> "+e.getLocalizedMessage());
                            return;
                        }
                    }
                    Timber.d(TAG+"---RUN_THREAD--- !mActive= "+mActive);
                    if (!mActive) {
                        // Exit the loop once this camera source is stopped or released.  We check
                        // this here, immediately after the wait() above, to handle the case where
                        // setActive(false) had been called, triggering the termination of this
                        // loop.
                        return;
                    }

                    // Hold onto the frame data locally, so that we can use this for detection
                    // below.  We need to clear mPendingFrameData to ensure that this buffer isn't
                    // recycled back to the camera before we are done using that data.
                    data = mPendingFrameData;

                    int width = mPreviewSize.getWidth();
                    int height = mPreviewSize.getHeight();
                    Timber.d(TAG+"CAMERA_PREV_WIDTH - "+width +" <> "+ height);
                    YuvImage yuv = new YuvImage(data.array(), ImageFormat.NV21, width, height, null);
                    ByteArrayOutputStream out = new ByteArrayOutputStream();
                    yuv.compressToJpeg(new Rect(0, 0, width, height), 50, out);
                    byte[] bytes = out.toByteArray();
                    Bitmap bm = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
                    try {
                        updateData(1, bm);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    mPendingFrameData = null;
                }

                mCamera.addCallbackBuffer(data.array());
            }
        }
    }

    public static void setCameraDisplayOrientation(Activity activity, int cameraId, android.hardware.Camera camera, Camera.Parameters parameters) {
        android.hardware.Camera.CameraInfo info =
                new android.hardware.Camera.CameraInfo();
        android.hardware.Camera.getCameraInfo(cameraId, info);
        int rotation = activity.getWindowManager().getDefaultDisplay()
                .getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
        }

        int result;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (info.orientation + degrees) % 360;
            result = (360 - result) % 360;  // compensate the mirror
        } else {  // back-facing
            result = (info.orientation - degrees + 360) % 360;
        }
        camera.setDisplayOrientation(result);
        parameters.setRotation(result);
    }

    private void captureImage() {
        if(null != mCamera) {
            setEnableOrDisabled(btnCaptureImage, false);
            progressBar.setVisibility(View.VISIBLE);
            mCamera.takePicture(null, null, this);
        }
    }

    private void setEnableOrDisabled(View e, boolean enabled)
    {
        e.setClickable(enabled);
        //e.setFocusable(enabled);
        //e.setFocusableInTouchMode(enabled);
    }

    private void setupImageDisplay() {

        if(null != mCamera) {
            mCamera.stopPreview();
            mCamera.startPreview();

            Bitmap bitmap = BitmapFactory.decodeByteArray(mCameraData, 0, mCameraData.length);
            //mCameraImage.setImageBitmap(bitmap);

            int screenWidth = getResources().getDisplayMetrics().widthPixels;
            int screenHeight = getResources().getDisplayMetrics().heightPixels;

            Timber.d(TAG+"rotation_orientation config.orientation - "+getResources().getConfiguration().orientation+" | "+Configuration.ORIENTATION_PORTRAIT);
            if (getResources().getConfiguration().orientation == Configuration.ORIENTATION_PORTRAIT) {
                // Notice that width and height are reversed
                Bitmap scaled = Bitmap.createScaledBitmap(bitmap, screenHeight, screenWidth, true);
                int w = scaled.getWidth();
                int h = scaled.getHeight();
                // Setting post rotate to 90
                Matrix mtx = new Matrix();

                boolean cameraFront = true;
                if (currentCameraId == Camera.CameraInfo.CAMERA_FACING_BACK) cameraFront = false;

                int CameraEyeValue = setPhotoOrientation(f, cameraFront == true ? 1 : 0); // CameraID = 1 : front 0:back
                if (cameraFront) { // As Front camera is Mirrored so Fliping the Orientation
                    if (CameraEyeValue == 270) {
                        mtx.postRotate(90);
                    } else if (CameraEyeValue == 90) {
                        mtx.postRotate(270);
                    }
                } else {
                    mtx.postRotate(CameraEyeValue); // CameraEyeValue is default to Display Rotation
                }

                bitmap = Bitmap.createBitmap(scaled, 0, 0, w, h, mtx, true);
            }

            String strImageName = "";

            if (Build.VERSION.SDK_INT >= 22) {
                strImageName = saveImageAPI23(bitmap);
            } else strImageName = saveImageAPIOlders(bitmap);

            listObject.add(new ListObject(strImageName, bitmap));
            previewCameraAdapter.notifyDataSetChanged();
            listView.smoothScrollToPosition(previewCameraAdapter.getItemCount() - 1);
            setEnableOrDisabled(btnCaptureImage, true);
            progressBar.setVisibility(View.GONE);
            tempPhoto.clear();
            int x = 0;
            for(ListObject o : listObject)
            {
                String s = (String) o.s1;
                String data2 = s + "|0|" + x;
                tempPhoto.put(""+x, data2);
                x++;
            }

        }
    }

    private int setPhotoOrientation(FragmentActivity activity, int cameraId) {
        android.hardware.Camera.CameraInfo info = new android.hardware.Camera.CameraInfo();
        android.hardware.Camera.getCameraInfo(cameraId, info);
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0:
                degrees = 0;
                break;
            case Surface.ROTATION_90:
                degrees = 90;
                break;
            case Surface.ROTATION_180:
                degrees = 180;
                break;
            case Surface.ROTATION_270:
                degrees = 270;
                break;
        }

        int result;
        // do something for phones running an SDK before lollipop
        if (info.facing == android.hardware.Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (info.orientation + degrees) % 360;
            result = (360 - result) % 360; // compensate the mirror
        } else { // back-facing
            Timber.d(TAG+"rotation_orientation info.orientation - "+info.orientation);
            result = (info.orientation - degrees + 360) % 360;
        }

        Timber.d(TAG+"rotation_orientation result - "+result);
        return result;
    }

    private String saveImageAPIOlders(Bitmap bitmap){
        String fname = FuncHelper.getRandomString("innosoft_gallery_")+ ".png";
        String folderImageName = "innosoft_gallery";
        String root = Environment.getExternalStorageDirectory().toString();
        File myDir = new File(root + folderImageName);
        if(!myDir.exists())myDir.mkdirs();
        File filedir = new File(myDir, fname);
        String imagePath = filedir.toString();

        File filename = new File(imagePath);

        try {
            FileOutputStream out = new FileOutputStream(filename);
            if (null == bitmap) {
            } else {
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
            }
            out.flush();
            out.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

        mPathResultImage.put(""+mkey, imagePath);
        mkey++;
        return imagePath;
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    private String saveImageAPI23(Bitmap bitmap){
        String fname = FuncHelper.getRandomString("innosoft_gallery_")+ ".png";
        String root = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM).toString();
        //String root = Environment.getExternalStorageDirectory().toString();
        File myDir = new File(root + "/Cameras/");
        if (!myDir.exists()) myDir.mkdirs();
        File filedir = new File(myDir, fname);
        String imagePath = filedir.toString();

        File filename = new File(imagePath);

        try {
            FileOutputStream out = new FileOutputStream(filename);
            if (null == bitmap) {
            } else {
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
            }
            out.flush();
            out.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

        mPathResultImage.put(""+mkey, imagePath);
        mkey++;
        return imagePath;
    }

    @Override
    public void onActivityResume() {
        Timber.d(TAG+"onActivityResume");
        //setCameras();
        //try {
        //    start(null);
        //} catch (IOException e) {
        //    e.printStackTrace();
        //}
    }

    @Override
    public void onActivityPause() {
        setCloseCamera();
    }

    @Override
    public void onActivityStop() {

    }

    @Override
    public void onActivityDestroy() {

    }

    @Override
    public void onActivityKeyDown(int keyCode, KeyEvent event) {

    }

    @Override
    public void onActivityFinish() {
        setCloseCamera();
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
        if (requestCode != DataStatic.PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6) {
            Log.d(TAG, "Got unexpected permission result: " + requestCode);
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            return;
        }

        if (grantResults.length != 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Timber.d(TAG+"Camera permission granted - initialize the camera source");
            // we have permission, so create the camerasource
            setCameras();
            return;
        }

        Timber.d(TAG+"Permission not granted: results len = " + grantResults.length +
                " Result code = " + (grantResults.length > 0 ? grantResults[0] : "(empty)"));

        finish();
    }

    @Override
    public void onActivityMResult(int requestCode, int resultCode, Intent data) {

    }

    @Override
    public void setAnimationOnOpenActivity(Integer firstAnim, Integer secondAnim) {
        overridePendingTransition(firstAnim, secondAnim);
    }

    @Override
    public void setAnimationOnCloseActivity(Integer firstAnim, Integer secondAnim) {
        overridePendingTransition(firstAnim, secondAnim);
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

    @Override
    public void close(ListObject data, int position) {

    }

    private void setCloseCamera()
    {
        //if (mCamera != null) {
        //    mCamera.release();
        //    mCamera = null;
        //}
        stop();
    }

    @Override
    public void onPictureTaken(byte[] data, Camera camera) {
        mCameraData = data;
        setupImageDisplay();
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        Timber.d(TAG+"surfaceChanged");
        /*
        if (mCamera != null) {
            try {
                Timber.d(TAG+"surfaceChanged mIsCapturing -> "+mIsCapturing);
                mCamera.setPreviewDisplay(holder);
                if (mIsCapturing) {
                    mCamera.startPreview();
                    setPrevCallBack();
                }
            } catch (IOException e) {
                Toast.makeText(f, "Cannot open cameras! -> "+e.getLocalizedMessage(), Toast.LENGTH_SHORT).show();
            }
        }
        */
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        Timber.d(TAG+"surfaceCreated");
        if(null == mCamera) {
            try {
                start(holder);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    private MultiBoxTracker tracker;
    private BlazeFace blazeFace;
    private FaceNet faceNet;
    private LibSVM svm;

    boolean updateData(int label, Bitmap bitmap) throws Exception {

        File dir = new File(FileUtils.ROOT);

        if (dir.isDirectory()) {
            if (dir.exists()) dir.delete();
            dir.mkdirs();

            AssetManager mgr = getAssets();
            FileUtils.copyAsset(mgr, FileUtils.DATA_FILE);
            FileUtils.copyAsset(mgr, FileUtils.MODEL_FILE);
            FileUtils.copyAsset(mgr, FileUtils.LABEL_FILE);
        }

        blazeFace = BlazeFace.create(getAssets());
        faceNet = FaceNet.create(getAssets());
        svm = LibSVM.getInstance();

        ArrayList<float[]> list = new ArrayList<>();
        boolean isFaceEmpty = true;
        List<RectF> faces = blazeFace.detect(bitmap);
        isFaceEmpty = faces.isEmpty();
        Timber.d(TAG+"FACES_DETECTED - FACES_IS_EMPTY - "+isFaceEmpty);
        Rect rect = new Rect();
        if (!isFaceEmpty) {
            Toast.makeText(f, "FACE_SUCCESS_DETECTED", Toast.LENGTH_SHORT).show();
            faces.get(0).round(rect);
            float[] emb_array = new float[FaceNet.EMBEDDING_SIZE];
            faceNet.getEmbeddings(bitmap, rect).get(emb_array);
            list.add(emb_array);
        }

        if (!isFaceEmpty)svm.train(label, list);

        return isFaceEmpty;
    }
}