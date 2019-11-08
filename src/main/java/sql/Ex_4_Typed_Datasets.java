package sql;

import java.io.Serializable;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;

import static rdd.util.SparkUtil.DATA_DIRECTORY;
import static rdd.util.SparkUtil.getSparkSession;

/**
 * A few typed operations.
 */
public class Ex_4_Typed_Datasets {
    public static void main(String[] args) {
        SparkSession spark = getSparkSession("Typed_Datasets");

        // Define Encoder for Bean class
        Encoder<StateNamesBean> personEncoder = Encoders.bean(StateNamesBean.class);
        Encoder<NameYearBean> nameYearEncoder = Encoders.bean(NameYearBean.class);

        Dataset<StateNamesBean> stateNames = spark.read().parquet(DATA_DIRECTORY + "/stateNames").as(personEncoder);
        stateNames.show();
        stateNames.printSchema();

        FilterFunction<StateNamesBean> filterFunction = value -> value.getGender().equals("F");
        MapFunction<StateNamesBean, NameYearBean> mapFunction = value -> new NameYearBean(value.name, value.count);

        Dataset result = stateNames
            .filter(filterFunction)
            .map(mapFunction, nameYearEncoder)
            .groupBy("name")
            .sum("count");

        result.show();
        result.explain(true);
    }

    public static class NameYearBean implements Serializable {
        private String name;

        private int count;

        public NameYearBean(String name, int count) {
            this.name = name;
            this.count = count;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public int getCount() {
            return count;
        }

        public void setCount(int count) {
            this.count = count;
        }
    }

    public static class StateNamesBean implements Serializable {
        private long id;

        private String name;

        private long year;

        private String gender;

        private String state;

        private Integer count;

        public long getId() {
            return id;
        }

        public void setId(long id) {
            this.id = id;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public long getYear() {
            return year;
        }

        public void setYear(long year) {
            this.year = year;
        }

        public String getGender() {
            return gender;
        }

        public void setGender(String gender) {
            this.gender = gender;
        }

        public String getState() {
            return state;
        }

        public void setState(String state) {
            this.state = state;
        }

        public Integer getCount() {
            return count;
        }

        public void setCount(Integer count) {
            this.count = count;
        }
    }
}
