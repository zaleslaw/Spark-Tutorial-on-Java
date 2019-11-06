package rdd;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

import static java.util.stream.Collectors.toList;
import static rdd.util.SparkUtil.DATA_DIRECTORY;
import static rdd.util.SparkUtil.getSparkSession;

/**
 * Parse StateNames file with RDD.
 */
public class Ex_8_Parse_CSV {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Parse_CSV");

        SparkContext sc = spark.sparkContext();
        JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

        // read from file
        JavaRDD<String> stateNamesCSV = jsc.textFile(DATA_DIRECTORY + "/StateNames.csv");

        // split / clean data
        JavaRDD<List<String>> headerAndRows = stateNamesCSV.map(line -> Arrays.asList(line.split(",")).stream().map(String::trim).collect(toList()));

        // get header
        List<String> header = headerAndRows.first();

        // filter out header (eh. just check if the first val matches the first header name)
        JavaRDD<List<String>> data = headerAndRows.filter(x -> !x.get(0).equals(header.get(0)));

        // splits to map (header/value pairs)
        JavaRDD<Map<String, String>> stateNames = data.map(splits -> zip(header, splits));

        // print top-5
        stateNames.take(5).forEach(System.out::println);

        // stateNames.collect // Easy to get java.lang.OutOfMemoryError: GC overhead limit exceeded

        // you should worry about all data transformations to rdd with schema
        stateNames
            .filter(e -> e.get("Name").equals("Anna") && Integer.valueOf(e.get("Count")) > 100)
            .take(5)
            .forEach(System.out::println);

        // the best way is here: try the DataFrames
    }

    private static Map<String, String> zip(List<String> header, List<String> splits) {
        if (header.size() != splits.size())
            throw new IllegalArgumentException("Both lists should be equal size.");

        Map<String, String> res = new HashMap<>();

        for (int i = 0; i < header.size(); i++)
            res.put(header.get(i), splits.get(i));

        return res;
    }
}
