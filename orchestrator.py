# Dropped, maybe in next iteration.


from enum import Enum

class DispatchType(Enum):
    TRAIN = 0
    EVALUATE = 1
    INFERENCE = 2


class ModelType(Enum):
    TFIDF_VECTORIZER = 0
    LOGISTIC_REGRESSION = 1
    LINEAR_SVC = 2
    CNN_TEXT_CLASSIFIER = 3
    LSTM_CLASSIFIER = 4


class Orchestrator:
    """
        Things start to take longer, and we also need to train the model under various
         configurations. For the future assignment, we might consider reusing parameters
         from previously trained models as starting points for further refining under
         different hyper-parameter configurations.

        For now, we want a few things:
            - a mechanism for determining how many models can be held in memory;
            - specifically for training though, how many models are feasibly trainable
               concurrently, based on the number of parameters they have and the
               available GPUs/CPUs;
            - a mechanism for dispatching multiple models to be trained, be evaluated,
               or do an inference, in a concurrent fashion;
            - a mechanism for timing the duration of a training (or evaluation, or inference);
            - save a model to a file (including some metadata about the model), or load
               it back into memory;
            - run the error analyses for various models.

        Problems:
            - for full reproducibility, the PRNG instances need to be separate. If by
               running tasks asynchronously, there ever happens that PRNG state leaks
               between jobs, then would affect reproducibility;
    """

    def __init__(self) -> None:
        pass


    def estimate_parallel_jobs():
        # It makes sense to dispatch multiple jobs if there are multiple GPUs,
        #  or if the GPU is rather large in comparison to our model's needs.

        if torch.cuda.is_available():
            count_GPUs: int = torch.cuda.device_count()
            free_vram, total_vram = torch.cuda.mem_get_info()

            # Each job-process needs to select its own CUDA device. Maybe have the orchestrator do this assignment and pass the decision in as a parameter in the cofiguration dict.
            return count_GPUs * 2

        else:
            # But if we don't have any GPU, just stick to a single dispatch
            #  at a time, synchronous execution essentially.
            return 1


    def dispatch(self,
                 model_type: ModelType,
                 dispatch_type: DispatchType,
                 configuration: dict[str, Any]
                 ) -> Future:
        # ::dispatch() will check whether specific (model_type x dispath_type) are
        #  implemented, and call another function to actually handle that.

        # This is becoming an actual job system.
        pass


    def train_tfidf_vectorizer(self) -> None:
        pass


# I am not sure what this should be. We might be dealing with: TfidfVectorizer, LogisticRegression, Linear SVC, CNNTextClassifier, LSTMClassifier, so uniformizing this parameter list might be a bit hard; maybe I should just use the model type with isinstance() in order to knkow how to read from a dict[str, Any] of parameters, so this can easily be very type-dynamic.