from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_extract, col
import sys

# Initialize Spark session
spark = SparkSession.builder.appName("GenerateImageMetadata").getOrCreate()
spark.sparkContext.setLogLevel("WARN")  # Reduce log verbosity

# Define GCS base path
BUCKET = "gs://retinopathy_dataset_bucket/RetinopathyDataset"
VERSIONS = ["augmented_resized_V2", "dr_unified_v2"]
SPLITS = ["train", "val", "test"]

# Loop through all dataset versions and splits
for version in VERSIONS:
    for split in SPLITS:
        input_path = f"{BUCKET}/{version}/{split}/*"
        output_path = f"{BUCKET}/etl_metadata/{version}_{split}_metadata.csv"

        try:
            print(f"Starting: {version}/{split}")

            # Load image files using Spark's binaryFile format
            df = spark.read.format("binaryFile") \
                .option("recursiveFileLookup", "true") \
                .option("pathGlobFilter", "*.jpg") \
                .load(input_path)

            # Drop raw image binary content
            df = df.drop("content")

            # Extract class label from path using regex
            df = df.withColumn("class", regexp_extract("path", rf"/{split}/(\d)/", 1).cast("int"))

            # Filter out small/corrupt files
            df = df.filter(col("length") > 10_000)

            # Repartition to optimize parallel writing
            df = df.repartition(10)

            # Save metadata as CSV to GCS
            df.select("path", "class", "length", "modificationTime") \
              .write.option("header", True).mode("overwrite").csv(output_path)

            print(f"Success: {output_path}")

        except Exception as e:
            print(f"Error with {input_path}: {e}", file=sys.stderr)

# Stop Spark session
spark.stop()
