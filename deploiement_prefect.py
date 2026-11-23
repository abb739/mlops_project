from prefect import serve

from pipeline_prefect import (
    flow_all,
    flow_code,
    flow_evaluate,
    flow_install,
    flow_predict,
    flow_train,
)

if __name__ == "__main__":
    serve(
        # runs every day at 02:00 AM
        flow_all.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
        ),
        flow_train.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        flow_evaluate.to_deployment(
            name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
        ),
        flow_predict.to_deployment(name="ml-pipeline-predict", tags=["mlops"]),
        flow_code.to_deployment(name="ml-pipeline-code", tags=["quality", "mlops"]),
        flow_install.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
    )
