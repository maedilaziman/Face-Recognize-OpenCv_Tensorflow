package com.maedi.soft.ino.recognize.face.recognizeupdate;

import android.content.Context;

import com.maedi.soft.ino.recognize.face.recognizeupdate.rcgutils.FileHelper;
import com.maedi.soft.ino.recognize.face.recognizeupdate.rcgutils.OneToOneMap;
import com.maedi.soft.ino.recognize.face.recognizeupdate.rcgutils.PreferencesHelper;

import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import timber.log.Timber;

public class SupportVectorMachine {
    private static final String TAG = "- OCVFaceDetect - ";
    private static final String STAG = "- ScanRecognizeFace - ";

    PreferencesHelper preferencesHelper;
    private FileHelper fh;
    private File trainingFile;
    private File predictionFile;
    private File testFile;
    private List<String> trainingList;
    private List<String> testList;
    private OneToOneMap<String, Integer> labelMap;
    private OneToOneMap<String, Integer> labelMapTest;
    private int method;

    public static final int TRAINING = 0;
    public static final int RECOGNITION = 1;

    public SupportVectorMachine(Context context, int method) {
        preferencesHelper = new PreferencesHelper(context);
        fh = new FileHelper();
        trainingFile = fh.createSvmTrainingFile();
        predictionFile = fh.createSvmPredictionFile();
        testFile = fh.createSvmTestFile();
        trainingList = new ArrayList<>();
        testList = new ArrayList<>();
        labelMap = new OneToOneMap<String, Integer>();
        labelMapTest = new OneToOneMap<String, Integer>();
        this.method = method;
        if(method == RECOGNITION){
            loadFromFile();
        }
    }

    public SupportVectorMachine(File trainingFile, File predictionFile){
        fh = new FileHelper();
        this.trainingFile = trainingFile;
        this.predictionFile = predictionFile;
        trainingList = new ArrayList<>();
    }

    // link jni library
    static {
        System.loadLibrary("jnilibsvm");
    }

    // connect the native functions
    private native void jniSvmTrain(String cmd);
    private native void jniSvmPredict(String cmd);

    public boolean train() {

        fh.saveStringList(trainingList, trainingFile);

        // linear kernel -t 0
        String svmTrainOptions = preferencesHelper.getSvmTrainOptions();
        String training = trainingFile.getAbsolutePath();
        String model = trainingFile.getAbsolutePath() + "_model";
        jniSvmTrain(svmTrainOptions + " " + training + " " + model);

        saveToFile();
        return true;
    }

    public boolean trainProbability(String svmTrainOptions) {
        fh.saveStringList(trainingList, trainingFile);
        String _svmTrainOptions = "";
        String training = trainingFile.getAbsolutePath();
        String model = trainingFile.getAbsolutePath() + "_model";
        jniSvmTrain(_svmTrainOptions + " -b 1" + " " + training + " " + model);

        return true;
    }

    public String recognize(Mat img, String expectedLabel) {

        try {
            FileWriter fw = new FileWriter(predictionFile, false);
            String line = imageToSvmString(img, expectedLabel);
            testList.add(line);
            fw.append(line);
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        String prediction = predictionFile.getAbsolutePath();
        Timber.d(STAG+"RECOGNIZE prediction -> "+prediction);
        String model = trainingFile.getAbsolutePath() + "_model";
        Timber.d(STAG+"RECOGNIZE model -> "+model);
        String output = predictionFile.getAbsolutePath() + "_output";
        Timber.d(STAG+"RECOGNIZE output -> "+output);
        jniSvmPredict(prediction + " " + model + " " + output);

        try {
            BufferedReader buf = new BufferedReader(new FileReader(output));
            String strBuf = buf.readLine();
            Timber.d(STAG+"RECOGNIZE Final output -> "+strBuf);
            int iLabel = null != strBuf ? Integer.valueOf(strBuf) : -1;
            buf.close();
            Timber.d(STAG+"RECOGNIZE #ILABEL# -> "+iLabel);
            return labelMap.getKey(iLabel);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public String setRecognizeProbability(Mat img)
    {
        String s = getSvmString(img);
        String recProbability = recognizeProbability(s);
        Timber.d(STAG+"RECOGNIZE PROBABILITY -> "+recProbability);
        return recProbability;
    }

    public String recognizeProbability(String svmString){
        try {
            FileWriter fw = new FileWriter(predictionFile, false);
            fw.append(String.valueOf(1) + svmString);
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        String prediction = predictionFile.getAbsolutePath();
        String model = trainingFile.getAbsolutePath() + "_model";
        String output = predictionFile.getAbsolutePath() + "_output";
        jniSvmPredict("-b 1 " + prediction + " " + model + " " + output);

        try {
            BufferedReader buf = new BufferedReader(new FileReader(output));
            // read header line
            String probability = buf.readLine() + "\n";
            // read content line
            probability = probability + buf.readLine();
            buf.close();
            Timber.d(STAG+"RECOGNIZE probability -> "+probability);
            return probability;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public void saveToFile() {
        if(method == TRAINING){
            fh.saveLabelMapToFile(fh.SVM_PATH, labelMap, "train");
        } else {
            fh.saveLabelMapToFile(fh.SVM_PATH, labelMapTest, "test");
        }
    }

    public void saveTestData(){
        fh.saveStringList(testList, testFile);
    }

    public void loadFromFile() {
        labelMap = fh.getLabelMapFromFile(fh.SVM_PATH);
    }

    public void addImage(Mat img, String label, boolean featuresAlreadyExtracted) {
        // Ignore featuresAlreadyExtracted because either SVM get the features from TensorFlow or Caffe, or it takes the image reshaping method (image itself)
        if(method == TRAINING){
            Timber.d(TAG+"TRAIN_FILE Label Name > "+label);
            trainingList.add(imageToSvmString(img, label));
        } else {
            testList.add(imageToSvmString(img, label));
        }
    }

    public void addImage(String svmString, String label) {
        trainingList.add(label + " " + svmString);
    }

    public Mat getFeatureVector(Mat img){
        return img.reshape(1,1);
    }

    private String imageToSvmString(Mat img, String label){
        int iLabel = 0;
        if(method != RECOGNITION) {
            if (method == TRAINING) {
                Timber.d(STAG + "TRAIN_FILE To String > " + label);
                if (labelMap.containsKey(label)) {
                    iLabel = labelMap.getValue(label);
                } else {
                    iLabel = labelMap.size() + 1;
                    Timber.d(STAG + "TRAIN_FILE I Label > " + iLabel);
                    labelMap.put(label, iLabel);
                }
            } else {
                Timber.d(STAG + "TEST_FILE I Label > " + label);
                if (labelMapTest.containsKey(label)) {
                    iLabel = labelMapTest.getValue(label);
                    Timber.d(STAG + "TEST_FILE I Label > " + label + " <> " + iLabel);
                } else {
                    iLabel = labelMapTest.size() + 1;
                    labelMapTest.put(label, iLabel);
                }
            }
        }

        String result = String.valueOf(iLabel);
        String resLast = result + getSvmString(img);
        Timber.d(STAG+"###############################TRAIN_FILE RES > "+result);
        return resLast;
    }

    public String getSvmString(Mat img){
        //img = getFeatureVector(img);
        String result = "";
        Timber.d(STAG+"TRAIN_FILE Get SVM String...");
        for (int i=0; i<img.cols(); i++){
            result = result + " " + i + ":" + img.get(0,i)[0];
        }
        Timber.d(STAG+"TRAIN_FILE Get SVM String Result> "+result);
        return result;
    }
}
