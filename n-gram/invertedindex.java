import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.StringTokenizer; 
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
public class invertedindex {
 
	public static class Map extends Mapper<Object, Text, Text, Text> {
        private String pattern = "[^a-zA-Z0-9-]";
        // 实现map函数
        // key=word+file_name
        // value=1
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // 获取文件名称
            FileSplit fileSplit = (FileSplit)context.getInputSplit();
            String filename = fileSplit.getPath().getName();
            // n-gram
            Integer n_gram = 2; 
            // 将每一行转化为一个String
            String line = value.toString();
            // 忽略大小写
            line.toLowerCase();
            // 将标点符号等字符用空格替换，这样仅剩单词
            line = line.replaceAll(pattern, " ");
            // 将String划分为一个个的单词
            String[] words = line.split("\\s+");
            List<String> words_list = new ArrayList<String>(Arrays.asList(words));
            words_list.remove("");
            // 将每一个n-gram初始化为词频为1，如果word相同，会传递给Reducer做进一步的操作
            for (int i=0;i<words_list.size()-n_gram;i++) {
                String temp = "";
                for(int j=0;j<n_gram;j++){    
                    temp += " " + words_list.get(i+j);
                }
                context.write(new Text(temp+','+filename), new Text("1"));
            }
        }
    }
 
    public static class Combine extends Reducer<Text, Text, Text, Text> {
        private Text info = new Text();
        // key=word
        // value=file_i+num_i
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // 统计词频
            int sum = 0;
            for (Text value : values) {
                sum += Integer.parseInt(value.toString());
            }
            int splitIndex = key.toString().indexOf(",");
            // 重新设置value
            info.set("("+key.toString().substring(splitIndex + 1) + "," + sum+")");
            // 重新设置key
            key.set(key.toString().substring(0, splitIndex));
            context.write(key, info);
        }
    }
 
	public static class Reduce extends Reducer<Text, Text, Text, Text> {
        private Text result = new Text();
        // key=word
        // value=(file_i,num_i),(file_j,num_j)
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // 录入value
            String fileList = new String();
            for (Text value : values) {
                fileList += value.toString() + ",";
            }
            result.set(fileList.substring(0,fileList.length()-1));
            context.write(key, result);
        }
    }

	public static void main(String[] args) throws Exception {
        // 以下部分为HadoopMapreduce主程序的写法，对照即可
        // 创建配置对象
        Configuration conf = new Configuration();
        // 创建Job对象
        Job job = new Job(conf, "Inverted Index");
        // 设置运行Job的类
        job.setJarByClass(invertedindex.class);
        // 设置Mapper类
        job.setMapperClass(Map.class);
        // 设置combiner类
        job.setCombinerClass(Combine.class);
        // 设置Reducer类
        job.setReducerClass(Reduce.class);
        // 设置Map输出类型
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        // 设置Reduce输出类型
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        // 设置输入和输出目录
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

