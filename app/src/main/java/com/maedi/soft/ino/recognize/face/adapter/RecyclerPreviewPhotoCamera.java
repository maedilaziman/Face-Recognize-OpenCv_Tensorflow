package com.maedi.soft.ino.recognize.face.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import com.maedi.soft.ino.recognize.face.R;
import com.maedi.soft.ino.recognize.face.model.ListObject;

import androidx.annotation.NonNull;
import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.RecyclerView;

public class RecyclerPreviewPhotoCamera extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    private Context context;
    private int layoutResourceId;
    private ListObject data = null;
    private String tag;
    private FragmentActivity f;

    public interface CommPreviewPhoto
    {
        void close(ListObject data, int position);
    }

    private CommPreviewPhoto listener;

    public RecyclerPreviewPhotoCamera(Context context, FragmentActivity f, int layoutResourceId, ListObject data, CommPreviewPhoto listener, String tag) {
        if(layoutResourceId != 0)this.layoutResourceId = layoutResourceId;
        this.context = context;
        this.data = data;
        this.tag = tag;
        this.f = f;
        this.listener = listener;
    }

    @Override
    public RecyclerPreviewPhotoCamera.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(layoutResourceId, parent, false);
        return new RecyclerPreviewPhotoCamera.ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder vvholder, int position) {

        if(data.size() > 0) {
            ViewHolder vholder = (ViewHolder) vvholder;
            ListObject lm = (ListObject) data.get(position);
            vholder.imageView.setImageBitmap(lm.bitmap1);

            vholder.imageClose.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    listener.close(data, position);
                }
            });
        }
    }

    public static class ViewHolder extends RecyclerView.ViewHolder{

        ImageView imageView, imageClose;

        public ViewHolder(View view) {
            super(view);
            this.imageView = (ImageView) view.findViewById(R.id.camera_image_view);
        }

        @Override
        public String toString() {
            return super.toString();
        }
    }

    @Override
    public int getItemViewType(int position) {
        return position;
    }

    @Override
    public int getItemCount() {
        return data.size();
    }

}