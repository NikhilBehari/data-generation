import numpy as np
import names


def generate_data_exams(num_students=10, num_exams=6, trend_up=5, exam_mean=75, exam_var=5):
    """Creates array with synthetic exam data

    :param num_students: number of students (default 10)
    :param num_exams: number of exams (default 6)
    :param trend_up: student improvement in exam scores (default 5)
    :param exam_mean: average exam score
    :param exam_var: variation in exam scores (default 5)
    :return: [names, [exam1_data, exam2_data, ...]]
    """
    exam_means = np.random.normal(exam_mean, exam_var, num_exams)
    exam_data = []
    temp_exam = []
    for student_ind in range(num_students):
        temp_exam.append(names.get_first_name())
    exam_data.append(temp_exam)
    temp_exam = []
    for exam_ind in range(num_exams):
        student_improvement = np.random.uniform(-2, trend_up/(num_exams-exam_ind), num_students)
        for student_ind in range(num_students):
            score = (np.random.normal((exam_means[exam_ind]+student_improvement[student_ind]), exam_var, 1)[0])
            if score > 100:
                score = 100
            temp_exam.append(round(score, 2))
        exam_data.append(temp_exam)
        temp_exam = []
    return np.asarray(exam_data)
