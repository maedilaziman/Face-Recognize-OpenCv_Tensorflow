package com.maedi.soft.ino.recognize.face.utils;

import android.Manifest;

public class DataStatic {

    public static final int PERMISSIONS_REQUEST_CAMERA = 1;

    public static final int PERMISSIONS_REQUEST_CAMERA_ABOVE6 = 2;

    public static final int PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6 = 17;

    public static final int PERMISSIONS_READ_EXTERNAL_STORAGE = 3;

    public static final int PERMISSIONS_READ_EXTERNAL_STORAGE_ABOVE6  = 4;

    public static final int PERMISSIONS_WRITE_EXTERNAL_STORAGE = 5;

    public static final int PERMISSIONS_WRITE_EXTERNAL_STORAGE_ABOVE6 = 6;

    public static final int REQUEST_INTENT_GALLERY = 105;

    public static final String EXTRA_INTENT_GALLERY = "INTENT-GALLERY";

    public static final String successSubmitNewData = "success_submit_new_data";

    public static final String[] GALLERY_PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    public static final String EXTRA_PATH_RESULT_IMG_CAMERA = "path_result_image_camera";

}
