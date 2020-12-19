package br.com.helpdev.facedetect;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.maedi.soft.ino.base.BuildActivity;
import com.maedi.soft.ino.base.func_interface.ActivityListener;
import com.maedi.soft.ino.base.store.MapDataParcelable;
import com.maedi.soft.ino.recognize.face.R;
import com.maedi.soft.ino.recognize.face.recognizeupdate.rcgutils.FileHelper;

import java.io.File;

import androidx.fragment.app.FragmentActivity;
import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;

public class ParenMainActivity extends BuildActivity<View> implements ActivityListener<Integer> {

    private final String TAG = this.getClass().getName() +"- ParentMainAct - ";

    private FragmentActivity f;

    @BindView(R.id.buttonAddUser)
    Button btnAddUser;

    @BindView(R.id.buttonScan)
    Button btnScan;

    @BindView(R.id.buttonDeleteParentFolder)
    Button btnDeleteParentFolder;

    @BindView(R.id.progressBar1)
    ProgressBar progressBar1;

    private boolean deleteDirectory(File directoryToBeDeleted) {
        File[] allContents = directoryToBeDeleted.listFiles();
        if (allContents != null) {
            for (File file : allContents) {
                deleteDirectory(file);
            }
        }
        return directoryToBeDeleted.delete();
    }

    @OnClick(R.id.buttonDeleteParentFolder)
    public void deleteParentFolder() {
        btnDeleteParentFolder.setEnabled(false);
        progressBar1.setVisibility(View.VISIBLE);
        File directoryToBeDeleted = new File(FileHelper.getFolderPath());
        deleteDirectory(directoryToBeDeleted);
        new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
            @Override
            public void run() {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        btnDeleteParentFolder.setEnabled(true);
                        progressBar1.setVisibility(View.GONE);
                        Toast.makeText(f, "Success deleted folder..", Toast.LENGTH_SHORT).show();
                    }
                });
            }
        }, 2000);
    }

    @OnClick(R.id.buttonAddUnknowUser)
    public void addUnknowUser() {
        File directoryUnknowUser = new File(FileHelper.TRAINING_PATH+ "/zzzUnknow/");
        if(directoryUnknowUser.exists())
        {
            Toast.makeText(f, "Directory unknow user is exist!, Please delete parent folder first", Toast.LENGTH_LONG).show();
            return;
        }
        Intent intent = new Intent(f, MainActivity.class);
        intent.putExtra("unknow_user", true);
        startActivity(intent);
    }

    @OnClick(R.id.buttonAddUser)
    public void addNewUser() {
        Intent intent = new Intent(f, MainActivity.class);
        intent.putExtra("unknow_user", false);
        startActivity(intent);
    }

    @OnClick(R.id.buttonScan)
    public void scanRecognizeFace() {
        Intent intent = new Intent(f, ScanRecognizeFace.class);
        startActivity(intent);
    }

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
        return R.layout.activity_parent_main;
    }

    @Override
    public ActivityListener createListenerForActivity() {
        return this;
    }

    @Override
    public void onCreateActivity(Bundle savedInstanceState) {
        f = this;
        ButterKnife.bind(this);
    }

    @Override
    public void onBuildActivityCreated() {
        init();
    }

    private void init()
    {

    }

    @Override
    public void onActivityResume() {

    }

    @Override
    public void onActivityPause() {

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
}
