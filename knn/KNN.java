import java.io.IOException;  
import java.util.StringTokenizer;  
import org.apache.hadoop.conf.Configuration;  
import org.apache.hadoop.fs.Path;  
import org.apache.hadoop.io.Text;  
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapreduce.Job;  
import org.apache.hadoop.mapreduce.Mapper;  
import org.apache.hadoop.mapreduce.Reducer;  
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;  
import org.apache.hadoop.mapreduce.lib.input.FileSplit;  
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;  
import org.apache.hadoop.util.GenericOptionsParser;  
  
import java.util.Collections;
import java.util.Comparator;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

public class KNN  
{  
      
    public static class Distance_Label {
        public float distance;//距离
        public String label;//标签
    }

    public static class TokenizerMapper extends Mapper<Object, Text, Text, Text> {  
        private ArrayList<ArrayList<Float>> test = new ArrayList<ArrayList<Float>> ();

        protected void setup(org.apache.hadoop.mapreduce.Mapper<Object, Text, Text, Text>.Context context) throws java.io.IOException, InterruptedException {
            // 设置测试集 ./test/iris_test
            FileSystem fs = FileSystem.get(context.getConfiguration());
            BufferedReader buff_reader = new BufferedReader(new InputStreamReader(fs.open(new Path(context.getConfiguration().get(
                    "org.niubility.learning.test", "./test/iris_test")))));
            int count = 0;
            String line = buff_reader.readLine();
            while (line != null) {
                String[] spilt_str = line.split(",");
                ArrayList<Float> testcase = new ArrayList<Float>();
                for (int i = 0; i < spilt_str.length-1; i++){
                    testcase.add(Float.parseFloat(spilt_str[i]));
                }
                test.add(testcase);
                line = buff_reader.readLine();
                count++;
            }
            buff_reader.close();
        }
    
        @Override  
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException  
        {
            // key 设置为训练数据行号
            context.setStatus(key.toString());
            // 拆分数据
            String[] spilt_str = value.toString().split(",");
            // 设置对应label
            String label = spilt_str[spilt_str.length - 1];   
            // 数据集逐行进行计算
            for (int i=0; i<test.size(); i++){
                ArrayList<Float> curr_test = test.get(i);
                double dis_sum = 0;
                // 欧式距离^2
                for(int j=0; j<curr_test.size(); j++){
                    dis_sum += (curr_test.get(j) - Float.parseFloat(spilt_str[j]))*(curr_test.get(j) - Float.parseFloat(spilt_str[j]));
                }
                // key:编号，value:(所有训练集距离,label)
                context.write(new Text(Integer.toString(i)), new Text(Double.toString(dis_sum)+","+label)); 
            }

        }  

    }  

    public static class KNNCombiner extends Reducer<Text, Text, Text, Text>  
    {  
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException  
        {  
            // key:编号，values:[ value(所有训练集距离,label) ]
            ArrayList<Distance_Label> dis_label_ar_l = new ArrayList<Distance_Label>();
            for (Text value : values){
                // value:(dis,label) -> split_str : [dis,label]
                String[] spilt_str = value.toString().split(",");
                Distance_Label tmp = new Distance_Label();
                tmp.label = spilt_str[1];
                tmp.distance = Float.parseFloat(spilt_str[0]);
                dis_label_ar_l.add(tmp);
            }
            // 排序 
            Collections.sort(dis_label_ar_l, new Comparator<Distance_Label>(){
                @Override
                // 升序
                public int compare(Distance_Label a, Distance_Label b){ 
                    if (a.distance > b.distance){
                        return 1;
                    }
                    return -1;
                }
            });
            // 设置K值
            final int K = 5; 
            /*
            * 加权KNN
            */
            // label & confidence
             HashMap<String, Double> label_confidence = new HashMap<String, Double>(); 
             for (int i=0; i<dis_label_ar_l.size() && i<K; i++){
                 String cur_l = dis_label_ar_l.get(i).label;
                 // 初始 label,0.0
                 if (!label_confidence.containsKey(cur_l)) label_confidence.put(cur_l, 0.0);
                 // label, 加权
                 label_confidence.put(cur_l, label_confidence.get(cur_l) + 1.0/(0.5+dis_label_ar_l.get(i).distance));
             }
             for (String l:label_confidence.keySet()){
                 context.write(key, new Text(Double.toString(label_confidence.get(l))+","+l));
             }
        }  
    }  

    public static class KNNReducer extends Reducer<Text, Text, Text, Text>  
    {  
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException  
        {  
            double max_dis = -1;
            String label = "";
            // 逐个进行比较
            for (Text val: values){
                String[] spilt_str = val.toString().split(","); 
                // 选择最大值最为结果
                if (Double.parseDouble(spilt_str[0]) > max_dis){
                    max_dis = Double.parseDouble(spilt_str[0]);
                    label = spilt_str[1];
                }
            }
            context.write(key, new Text(label));
        }
    }  
      
    public static void main(String[] args) throws Exception  
    {  
        Configuration conf=new Configuration();  
        if(args.length!=2)  
        {  
            System.out.println("args: KNN <in> <out>");  
            System.exit(2);  
        }  
        Path inputPath=new Path(args[0]);  
        Path outputPath=new Path(args[1]);  
        outputPath.getFileSystem(conf).delete(outputPath, true);  
          
        Job job=Job.getInstance(conf, "KNN");  
        job.setJarByClass(KNN.class);  
          
        job.setMapperClass(TokenizerMapper.class);  
        job.setMapOutputKeyClass(Text.class);  
        job.setMapOutputValueClass(Text.class);  
           
        job.setCombinerClass(KNNCombiner.class);  
          
        job.setReducerClass(KNNReducer.class);          
        job.setOutputKeyClass(Text.class);  
        job.setOutputValueClass(Text.class);  
          
        FileInputFormat.addInputPath(job, inputPath);  
        FileOutputFormat.setOutputPath(job, outputPath);  
          
        System.exit(job.waitForCompletion(true)? 0:1);  
    }  
      
}  
