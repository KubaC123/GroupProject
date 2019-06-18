package com.example.placerecogniser;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

public class PlaceList extends AppCompatActivity {

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_place_list);


        FloatingActionButton deleteAll = findViewById(R.id.deleteAll);
        deleteAll.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                findViewById(R.id.textView1).setVisibility(View.INVISIBLE);
                findViewById(R.id.textView2).setVisibility(View.INVISIBLE);
                findViewById(R.id.textView3).setVisibility(View.INVISIBLE);
                findViewById(R.id.textView4).setVisibility(View.INVISIBLE);
                findViewById(R.id.textView5).setVisibility(View.INVISIBLE);
                findViewById(R.id.textView6).setVisibility(View.INVISIBLE);
            }
        });
    }


}
