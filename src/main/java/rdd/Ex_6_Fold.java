package rdd;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

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
        val codeRows = sc.makeRDD(List(("Ivan", 240), ("Elena", -15), ("Petr", 39),("Elena", 290)))

        val zeroCoder = ("zeroCoder", 0);

        val greatContributor = codeRows.fold(zeroCoder) (
            (acc, coder) =>{
            if (acc._2 < Math.abs(coder._2))
                coder
            else
                acc
        })

        println("Developer with maximum contribution is " + greatContributor)

        // Step-2: Group code rows by skill

        val codeRowsBySkill = sc.makeRDD(List(("Java", ("Ivan", 240)), ("Java", ("Elena", -15)),("PHP", ("Petr", 39)),
        ("PHP", ("Elena", 290))))

        val maxBySkill = codeRowsBySkill.foldByKey(zeroCoder) (
            (acc, coder) =>{
            if (acc._2 > Math.abs(coder._2))
                acc
            else
                coder
        })

        println("Greatest contributor by skill are " + maxBySkill.collect().toList)
    }
}
