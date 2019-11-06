package rdd;

import java.util.Arrays;
import java.util.List;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import static rdd.util.SparkUtil.DATA_DIRECTORY;
import static rdd.util.SparkUtil.getSparkSession;

/**
 * Save PairRDD to SequenceFile and load data back.
 */
public class Ex_4_Save_Load_PairRDD_To_SequenceFile {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Save_Load_PairRDD_To_SequenceFile");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        List<Tuple2<String, String>> profileData = Arrays.asList(new Tuple2<>("Ivan", "Java"), new Tuple2<>("Elena", "Scala"), new Tuple2<>("Petr", "Scala"));
        JavaPairRDD<String, String> programmerProfiles = jsc.parallelizePairs(profileData);
        programmerProfiles
            .mapToPair(row -> new Tuple2<>(new Text(row._1), new Text(row._2)))
            .saveAsNewAPIHadoopFile(DATA_DIRECTORY + "/profiles", Text.class, Text.class, SequenceFileOutputFormat.class);

        // Read and parse data
        PairFunction<Tuple2<Text, Text>, String, String> castTypesFunction = row -> new Tuple2<>(row._1.toString(), row._2.toString());
        JavaPairRDD<String, String> profiles = jsc.sequenceFile(DATA_DIRECTORY + "/profiles", Text.class, Text.class).mapToPair(
            castTypesFunction
        );

        profiles.collect().forEach(System.out::println);
    }
}
