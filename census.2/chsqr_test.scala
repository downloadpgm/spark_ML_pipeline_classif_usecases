import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics

def chsqr_test(df: DataFrame, categ_col_name: String, label_col_name:String ):Unit = {
  val categ_name = categ_col_name
  val label_name = label_col_name
  
  val str1 = new StringIndexer().setInputCol(label_name).setOutputCol(label_name+"idx")
  val df1 = str1.fit(df).transform(df)
  val str2 = new StringIndexer().setInputCol(categ_name).setOutputCol(categ_name+"idx")
  val df2 = str2.fit(df1).transform(df1)

  val obs = df2.select(col(label_name+"idx"),col(categ_name+"idx")).rdd.map( x => {
    val l = x.getDouble(0)
    val f = x.getDouble(1)
    LabeledPoint(l,Vectors.dense(f)) })

  val featureTestResults = Statistics.chiSqTest(obs)
  featureTestResults.foreach(println)
  println("           " + categ_name + "idx")
  df2.groupBy(label_name+"idx").pivot(categ_name+"idx").count.show
}
