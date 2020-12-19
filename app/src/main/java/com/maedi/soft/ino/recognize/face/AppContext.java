package com.maedi.soft.ino.recognize.face;

import android.app.Application;

import com.maedi.soft.ino.base.annotation.BuilderAnnotations.HeaderService;
import com.maedi.soft.ino.base.annotation.BuilderAnnotations.SetAnalytics;

import timber.log.Timber;

public class AppContext extends Application {

    //key and value is your api key - default is empty
    //baseUrl is your base url server - this baseUrl cannot empty
    //timeout is connection timeout, default is 60 second (60000)
    @HeaderService(key="", value="", baseUrl="https://jsonplaceholder.typicode.com/", timeout = "60000")
    String dataApiKey;

    //default set analytics is true
    @SetAnalytics(enabled = false)
    public void setAnalytics(String valueTag){
    }

    @Override
    public void onCreate() {
        super.onCreate();
        if (BuildConfig.DEBUG) {
            Timber.plant(new Timber.DebugTree());
        }
    }
}
