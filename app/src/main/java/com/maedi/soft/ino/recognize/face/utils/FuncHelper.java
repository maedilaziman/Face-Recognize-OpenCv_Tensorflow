package com.maedi.soft.ino.recognize.face.utils;

import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.os.Build;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;

import com.maedi.soft.ino.recognize.face.R;

import java.security.SecureRandom;
import java.util.List;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;
import pub.devrel.easypermissions.EasyPermissions;

public class FuncHelper {

    public static String getRandomString(String param){
        long currentTimeMillis = System.currentTimeMillis();
        SecureRandom random = new SecureRandom();

        char[] CHARSET_AZ_09 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".toCharArray();
        char[] result = new char[15];
        for (int i = 0; i < result.length; i++) {
            int randomCharIndex = random.nextInt(CHARSET_AZ_09.length);
            result[i] = CHARSET_AZ_09[randomCharIndex];
        }

        String resRandom = param+String.valueOf(currentTimeMillis)+new String(result);

        return resRandom;
    }

    public static int getPictureSizeIndexForHeight(List<Camera.Size> sizeList, int height) {
        int chosenHeight = -1;
        for(int i=0; i<sizeList.size(); i++) {
            if(sizeList.get(i).height < height) {
                chosenHeight = i-1;
                if(chosenHeight==-1)
                    chosenHeight = 0;
                break;
            }
        }
        return chosenHeight;
    }

    public static void cameraRequestPermissionsResult(FragmentActivity f, int requestCode, int[] grantResults) {
        if (requestCode == DataStatic.PERMISSIONS_REQUEST_CAMERA) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {}
            else {
                f.finish();
            }
        }else if (requestCode == DataStatic.PERMISSIONS_REQUEST_CAMERA_ABOVE6) {
            if (EasyPermissions.hasPermissions(f, DataStatic.GALLERY_PERMISSIONS)) {}
            else {
                f.finish();
            }
        }
    }

    public static void showOverlay(FragmentActivity f, View view, View view2)
    {
        showHideOverlayAnimation(f, view, view2, R.anim.overlay_in_from_top, true, 1);
    }

    public static void hideOverlay(final FragmentActivity f, final View view, final View view2)
    {
        showHideOverlayAnimation(f, view, view2, R.anim.overlay_out_to_top_fast1, false, 1);
    }

    public static void showHideOverlayAnimation(final FragmentActivity f, final View view, View view2, int animStyle, final boolean show, final int onAnimStart){
        if(show)view.setVisibility(View.VISIBLE);
        //if(null != view2)view2.setVisibility(View.VISIBLE);
        Animation animation = AnimationUtils.loadAnimation(f, animStyle);
        //animation.setDuration(500);
        animation.setAnimationListener(new Animation.AnimationListener() {
            @Override
            public void onAnimationStart(Animation animation) {
                // TODO Auto-generated method stub
                if(onAnimStart == 1)
                {
                    if(!show)view.setVisibility(View.GONE);
                }
            }

            @Override
            public void onAnimationRepeat(Animation animation) {
                // TODO Auto-generated method stub
            }

            @Override
            public void onAnimationEnd(Animation animation) {
                // TODO Auto-generated method stub
                if(onAnimStart == 0)
                {
                    if(!show)view.setVisibility(View.GONE);
                }
            }
        });
        view.startAnimation(animation);
    }

    public static boolean hasAPI_LEVEL24_ANDROID_7_Above() {
        return Build.VERSION.SDK_INT >= 24; //API Level 7
    }

    public static void CameraPermission_API_LEVEL24_ANDROID_7_Above(FragmentActivity f){

        EasyPermissions.requestPermissions(f, "CAMERA",
                DataStatic.PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6, DataStatic.GALLERY_PERMISSIONS);
    }

    public static void CameraPermission(FragmentActivity f) {
        if (ContextCompat.checkSelfPermission(f, DataStatic.GALLERY_PERMISSIONS[0]) != PackageManager.PERMISSION_GRANTED) {

            if (ActivityCompat.shouldShowRequestPermissionRationale(f, DataStatic.GALLERY_PERMISSIONS[0])) {

                ActivityCompat.requestPermissions(f,
                        DataStatic.GALLERY_PERMISSIONS,
                        DataStatic.PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6);

            } else {

                ActivityCompat.requestPermissions(f,
                        DataStatic.GALLERY_PERMISSIONS,
                        DataStatic.PERMISSION_REQUEST_ACCESS_CAMERA_ABOVE6);
            }
        }
    }
}
