import dspy
import dsp

from .predict import Predict


class Retry(Predict):
    def __init__(self, module):
        super().__init__(module.signature)
        self.module = module
        self.original_signature = module.signature.signature
        self.original_forward = module.forward
        self.new_signature = self._create_new_signature(self.original_signature)

    def _create_new_signature(self, original_signature):
        extended_signature = {}
        input_fields = original_signature.input_fields()
        output_fields = original_signature.output_fields()
        modified_output_fields = {}

        # @manish: TODO: verify this and remove past_outputs from traces
        # for key, value in output_fields.items():
        #     modified_output_fields[f"past_{key}"] = dspy.OutputField(
        #         prefix="Past " + value.prefix, desc=value.desc, format=value.format
        #     )

        extended_signature.update(input_fields)
        # extended_signature.update(modified_output_fields)

        extended_signature["feedback"] = dspy.InputField(
            prefix="Instructions:",
            desc="Some instructions you must satisfy",
            format=str,
        )

        extended_signature["traces"] = dspy.InputField(
            prefix="Traces:",
            desc="traces of failures from your past attempts",
            format=dsp.passages2text,
        )

        extended_signature.update(output_fields)

        return extended_signature

    def forward(self, *args, **kwargs):
        kwargs["signature"] = self.new_signature
        return self.original_forward(**kwargs)
