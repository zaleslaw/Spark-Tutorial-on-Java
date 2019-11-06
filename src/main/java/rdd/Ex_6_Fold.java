package rdd;

import java.util.Arrays;
import java.util.List;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import static rdd.util.SparkUtil.getSparkSession;
import static rdd.util.SparkUtil.println;

/**
 * Fold operation example.
 */
public class Ex_6_Fold {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Fold");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        // Step-1: Find greatest contributor
        List<Tuple2<String, Integer>> data = Arrays.asList(new Tuple2<>("Elena", -15), new Tuple2<>("Petr", 39), new Tuple2<>("Elena", 290));
        JavaPairRDD<String, Integer> codeRows = jsc.parallelizePairs(data);

        Tuple2<String, Integer> zeroCoder = new Tuple2<>("zeroCoder", 0);

        Tuple2<String, Integer> greatContributor = codeRows.fold(zeroCoder,
            (acc, coder) -> {
                if (acc._2 < Math.abs(coder._2))
                    return coder;
                else
                    return acc;
            });

        println("Developer with maximum contribution is " + greatContributor);

        // Step-2: Group code rows by skill
        List<Tuple2<String, Tuple2<String, Integer>>> codeRowsBySkillData = Arrays
            .asList(
                new Tuple2<>("Java", new Tuple2<>("Ivan", 240)),
                new Tuple2<>("Java", new Tuple2<>("Elena", -15)),
                new Tuple2<>("PHP", new Tuple2<>("Petr", 39))
            );

        JavaPairRDD<String, Tuple2<String, Integer>> codeRowsBySkill = jsc.parallelizePairs(codeRowsBySkillData);

        JavaPairRDD<String, Tuple2<String, Integer>> maxBySkill = codeRowsBySkill.foldByKey(zeroCoder,
            (acc, coder) -> {
                if (acc._2 > Math.abs(coder._2))
                    return acc;
                else
                    return coder;
            });

        println("Greatest contributor by skill are " + Arrays.deepToString(maxBySkill.collect().toArray()));
    }
}
