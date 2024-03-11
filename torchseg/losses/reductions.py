
#: NONE_REDUCTION mode: the output will be summed
MEAN_REDUCTION: str = 'mean'

#: SUM_REDUCTION mode: the output will be summed
SUM_REDUCTION: str = 'sum'

#: NONE_REDUCTION mode: no reduction will be applied
NONE_REDUCTION: str = 'none'


class LossReduction:
    """
    Class to represent different modes of reduction.
    - MEAN: the sum of the output will be divided by the number of elements in the output;
    - SUM: the output will be summed;
    - NONE: no reduction will be applied
    """

    MEAN = "mean"
    SUM = "sum"
    NONE = "none"

    @staticmethod
    def available_reductions():
        return [LossReduction.MEAN, LossReduction.SUM, LossReduction.NONE]



