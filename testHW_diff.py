import numpy as np
import cv2 as cv
import os
import evaluation as eval


###############################################################
##### This code has been tested in Python 3.6 environment #####
###############################################################

def main():
    ##### Set threshold
    threshold1 = 8.5

    ##### Set path
    input_path = './input'  # input path
    gt_path = './groundtruth'  # groundtruth path
    result_path = './result'  # result path

    ##### load input
    input = [img for img in sorted(os.listdir(input_path)) if img.endswith(".jpg")]

    ##### first frame and first background
    frame_current = cv.imread(os.path.join(input_path, input[0]))
    frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
    frame_prev_gray = frame_current_gray

    # 모든 프레임에 대한 median을 계산하는 것은 오랜 시간이 걸려 샘플링 후 median을 계산
    frame_median = np.ndarray(shape=frame_current_gray.shape)

    # 계산을 위해 shape를 저장
    n = frame_current_gray.shape[0]
    m = frame_current_gray.shape[1]

    # median 값 계산을 위해 n*m개의 딕셔너리를 생성함
    # 딕셔너리의 key는 색깔, value는 counting 용도
    frame_median_dict = [[{} for j in range(m)] for i in range(n)]

    # 50 프레임 마다 하나의 샘플링을 통해 메디안을 계산
    for image_idx in range(0, len(input), 50):
        if image_idx >= len(input) - 1:
            break

        frame = cv.imread(os.path.join(input_path, input[image_idx + 1]))
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float64)

        for i in range(n):
            for j in range(m):
                col = frame_gray[i][j]
                if col in frame_median_dict[i][j].keys():
                    frame_median_dict[i][j][col] = frame_median_dict[i][j][col] + 1
                else:
                    frame_median_dict[i][j][col] = 1

    # keys를 정렬 후 딕셔너리를 keys로 순회하며 median_count에서 key에 해당하는 value만큼 빼줌
    # 초기 median_count가 median이 될 index이므로 0이거나 음수가 되면 그 값이 median
    for i in range(n):
        for j in range(m):
            median_count = int(int(len(input) / 2) / 50)
            keys = list(frame_median_dict[i][j].keys())
            keys.sort()
            for k in keys:
                median_count = median_count - frame_median_dict[i][j][k]
                if median_count <= 0:
                    frame_median[i][j] = k
                    break

    diff_temp = np.abs(frame_current_gray - frame_median).astype(np.float64)

    ##### background substraction
    for image_idx in range(len(input)):

        print(image_idx)

        ##### calculate foreground region
        diff = frame_current_gray - frame_median
        diff_abs = np.abs(diff).astype(np.float64)

        # diff_prev = frame_current_gray - frame_prev_gray
        # diff_abs2 = np.abs(diff_prev).astype(np.float64)

        ##### make mask by applying threshold
        frame_diff = np.where(diff_abs > threshold1, 1.0, 0.0)

        # r = 0.005
        # frame_diff = (1 - r) * frame_diff1 + r * frame_diff2

        ##### apply mask to current frame
        current_gray_masked = np.multiply(frame_current_gray, frame_diff)
        current_gray_masked_mk2 = np.where(current_gray_masked > 0, 255.0, 0.0)

        ##### result
        result = current_gray_masked_mk2.astype(np.uint8)

        result = cv.medianBlur(result, 13)
        kernel = np.ones((9, 9), np.uint8)
        result = cv.morphologyEx(result, cv.MORPH_OPEN, kernel=np.ones((9, 9), np.uint8))
        result = cv.morphologyEx(result, cv.MORPH_CLOSE, kernel=np.ones((9, 9), np.uint8))

        cv.imshow('result', result)

        ##### make result file
        ##### Please don't modify path
        cv.imwrite(os.path.join(result_path, 'result%06d.png' % (image_idx + 1)), result)

        ##### end of input
        if image_idx == len(input) - 1:
            break

        ##### read next frame
        frame_current = cv.imread(os.path.join(input_path, input[image_idx + 1]))
        frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)

        ##### If you want to stop, press ESC key
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    ##### evaluation result
    eval.cal_result(gt_path, result_path)


if __name__ == '__main__':
    main()
