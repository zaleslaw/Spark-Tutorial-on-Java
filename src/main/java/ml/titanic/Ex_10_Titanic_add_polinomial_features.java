package ml.titanic;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.PolynomialExpansion;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Add polynomial features. Accuracy = 0.222. Not so good, but maybe we can choose a subset of best features?
 */
public class Ex_10_Titanic_add_polinomial_features {
    public static void main(String[] args) {
        SparkSession spark = TitanicUtils.getSparkSession();

        Dataset<Row> passengers = TitanicUtils.readPassengersWithCasting(spark)
            .select("survived", "pclass", "sibsp", "parch", "sex", "embarked", "age", "fare");

        Dataset<Row>[] split = passengers.randomSplit(new double[] {0.7, 0.3}, 12345);
        Dataset<Row> training = split[0].cache();
        Dataset<Row> test = split[1].cache();

        StringIndexer sexIndexer = new StringIndexer()
            .setInputCol("sex")
            .setOutputCol("sexIndexed")
            .setHandleInvalid("keep"); // special mode to create special double value for null values

        StringIndexer embarkedIndexer = new StringIndexer()
            .setInputCol("embarked")
            .setOutputCol("embarkedIndexed")
            .setHandleInvalid("keep"); // special mode to create special double value for null values

        Imputer imputer = new Imputer()
            .setInputCols(new String[]{"pclass", "sibsp", "parch", "age", "fare"})
            .setOutputCols(new String[]{"pclass_imputed", "sibsp_imputed", "parch_imputed", "age_imputed", "fare_imputed"})
            .setStrategy("mean");

        VectorAssembler assembler = new VectorAssembler()
            .setInputCols(new String[]{"pclass_imputed", "sibsp_imputed", "parch_imputed", "age_imputed", "fare_imputed", "sexIndexed", "embarkedIndexed"})
            .setOutputCol("features");

        PolynomialExpansion polyExpansion = new PolynomialExpansion()
            .setInputCol("features")
            .setOutputCol("polyFeatures")
            .setDegree(2);

        MinMaxScaler scaler = new MinMaxScaler() // new MaxAbsScaler()
            .setInputCol("polyFeatures")
            .setOutputCol("unnorm_features");

        Normalizer normalizer = new Normalizer()
            .setInputCol("unnorm_features")
            .setOutputCol("norm_features")
            .setP(1.0);

        DecisionTreeClassifier trainer = new DecisionTreeClassifier()
                .setLabelCol("survived")
                .setFeaturesCol("norm_features");

        Pipeline pipeline = new Pipeline()
            .setStages(new PipelineStage[]{sexIndexer, embarkedIndexer, imputer, assembler, polyExpansion, scaler, normalizer, new TitanicUtils.Printer(), trainer});

        PipelineModel model = pipeline.fit(training);

        Dataset<Row> rawPredictions = model.transform(test);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("survived")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(rawPredictions);
        System.out.println("Test Error = " + (1.0 - accuracy));
    }
}
