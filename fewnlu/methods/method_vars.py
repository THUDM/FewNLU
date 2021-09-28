# methods
from methods.adapet.adapet_model import AdaPETModel
from methods.lm_training.lm_training_model import LMTrainingModel
# from methods.noisy_student.noisy_student_model import NoisyStudentModel
from methods.pet.pet_model import PetModel
from methods.ptuning.ptuning_model import ContinuousPromptModel
from methods.sequence_classifier.cls_model import SequenceClassifierModel

METHOD_CLASSES = {
    'sequence_classifier': SequenceClassifierModel,
    'pet': PetModel,
    'ptuning': ContinuousPromptModel,
    'adapet': AdaPETModel,
    # 'lm_training': LMTrainingModel,
    # 'noisy_student': NoisyStudentModel,
    # 'ipet': PetModel,
}

ARCH_METHOD_CLASSES=['default','ipet','noisy_student']