import cv2
import numpy as np


def filter_contours(stats, area_th, width_th, height_th):
    filtered_idx = []
    for i, stat in enumerate(stats):
        x, y, width, height, area = stat
        if area > area_th and width > width_th and height > height_th:
            filtered_idx.append(i)
    return filtered_idx

def draw_empty_spaces_between_contours(image, stats, filtered_idx, min_space_th, max_space_th):
    right_edges_x = [stats[idx, cv2.CC_STAT_LEFT] + stats[idx, cv2.CC_STAT_WIDTH] for idx in filtered_idx]
    left_edges_x = [stats[idx, cv2.CC_STAT_LEFT] for idx in filtered_idx]

    top_edges_y = [stats[idx, cv2.CC_STAT_TOP] for idx in filtered_idx]
    bottom_edges_y = [stats[idx, cv2.CC_STAT_TOP] + stats[idx, cv2.CC_STAT_HEIGHT] for idx in filtered_idx]

    for i in range(1, len(filtered_idx) - 1):
        
        space_width = left_edges_x[i + 1] - right_edges_x[i]
        top_y = min(top_edges_y[i], top_edges_y[i + 1])
        bottom_y = max(bottom_edges_y[i], bottom_edges_y[i + 1])

        if min_space_th < space_width <= max_space_th:
            cv2.rectangle(image, (right_edges_x[i], top_y), (left_edges_x[i + 1], bottom_y), (0, 255, 0), 2)
            cv2.putText(image, f"W: {space_width}", (right_edges_x[i] + 5, top_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return image

def sign(x):
    return int(x > 0) - int(x < 0)

def find_w_center(lst, dist_th, height_th):
    prev_slope_sign = sign(lst[1] - lst[0])
    changes = []
    for i in range(2, len(lst)):
        current_slope_sign = sign(lst[i] - lst[i-1])
        if current_slope_sign != 0 and current_slope_sign != prev_slope_sign:
            changes.append(i-1)
            prev_slope_sign = current_slope_sign
    changes.append(len(lst)-1)

    w_centers = []
    for i in range(1, len(changes) - 1):
        if lst[changes[i-1]] > lst[changes[i]] < lst[changes[i+1]]:
            flat_start = changes[i]
            flat_end = changes[i]
            while flat_start > changes[i-1] and lst[flat_start-1] == lst[changes[i]]:
                flat_start -= 1
            while flat_end < changes[i+1] and lst[flat_end+1] == lst[changes[i]]:
                flat_end += 1
            w_center = flat_start + (flat_end - flat_start) // 2

            change_amount = abs(lst[changes[i-1]] - lst[changes[i]]) + abs(lst[changes[i+1]] - lst[changes[i]])
            if change_amount <= height_th:  # Check if height difference exceeds 3
                continue

            if w_centers and (w_center - w_centers[-1][0]) < dist_th:  # Check if x distance exceeds dist_th
                continue

            w_centers.append((w_center, change_amount))

    return [(wc[0], wc[1]) for wc in w_centers if wc[1] > 3]  # Filter points exceeding height threshold

def find_m_center(lst, dist_th, height_th):
    prev_slope_sign = sign(lst[1] - lst[0])
    changes = []
    for i in range(2, len(lst)):
        current_slope_sign = sign(lst[i] - lst[i-1])
        if current_slope_sign != 0 and current_slope_sign != prev_slope_sign:
            changes.append(i-1)
            prev_slope_sign = current_slope_sign
    changes.append(len(lst)-1)

    m_centers = []
    for i in range(1, len(changes) - 1):
        if lst[changes[i-1]] < lst[changes[i]] > lst[changes[i+1]] and lst[changes[i]] > height_th:
            flat_start = changes[i]
            flat_end = changes[i]
            while flat_start > changes[i-1] and lst[flat_start-1] == lst[changes[i]]:
                flat_start -= 1
            while flat_end < changes[i+1] and lst[flat_end+1] == lst[changes[i]]:
                flat_end += 1
            m_center = flat_start + (flat_end - flat_start) // 2

            change_amount = abs(lst[changes[i-1]] - lst[changes[i]]) + abs(lst[changes[i+1]] - lst[changes[i]])
            if change_amount <= height_th:  # Check if height difference exceeds 3
                continue

            if m_centers and (m_center - m_centers[-1][0]) < dist_th:  # Check if x distance exceeds dist_th
                continue

            m_centers.append((m_center, change_amount))

    return [(mc[0], mc[1]) for mc in m_centers if mc[1] > 3 and lst[mc[0]] > height_th]  # Filter points exceeding both thresholds

def calculate_boundaries(contour, image_shape):
    """경계선의 최상단과 최하단 계산"""
    lstTopBoundary = [image_shape[0]] * image_shape[1]
    lstBotBoundary = [-1] * image_shape[1]

    iStartX = image_shape[1]
    iEndX = -1
   
    for pt in contour:
        x = pt[0][0]
        y = pt[0][1]
       
        if y < lstTopBoundary[x]:
            lstTopBoundary[x] = y
        if y > lstBotBoundary[x]:
            lstBotBoundary[x] = y

        iStartX = min(iStartX, x)
        iEndX = max(iEndX, x)

    return iStartX, iEndX, lstTopBoundary, lstBotBoundary


def draw_boundaries(image, iStartX, iEndX, lstTopBoundary, lstBotBoundary):
    """계산된 경계선을 이미지에 그리기"""
    for j in range(iStartX, iEndX):
        image[lstTopBoundary[j], j] = (0,0,255) # 최상단 경계선 빨간색으로 그리기
        image[lstBotBoundary[j], j] = (255,255,0) # 최하단 경계선 노란색으로 그리기
    return image

def process_image(image_path, dist_th, height_th):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image load failed.")
        return
    
    th = 100
    _, binImg = cv2.threshold(img, th, 255, cv2.THRESH_BINARY_INV)
 
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kernel)
 
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing)
    output_check = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    output_check_temp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    output_check_peak = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # for i in range(1, num_labels): # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
    #     (x, y, w, h, area) = stats[i]

    #     if area < 20:
    #         continue
    #     cv2.rectangle(output_check, (x, y, w, h), (0, 255, 255))

    # cv2.imshow('Output with Empty Spaces', output_check)
    # cv2.imwrite("testdfdd.bmp", output_check)

    area_th = 20
    width_th = 1    
    height_contour_th = 1

    min_space_th = 30
    max_space_th = 200

    filtered_idx = filter_contours(stats, area_th, width_th, height_contour_th)
    filtered_idx_ = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= area_th]
    
    # x 방향으로 contours 정렬
    sorted_idx = sorted(filtered_idx_, key=lambda i: stats[i, cv2.CC_STAT_LEFT])

    '''contours 후보군들 간의 distance 측정 하여 draw / width가 설정한 th를 넘어야 distance 측정 가능하게끔 수정'''
    for i in range(len(sorted_idx) - 1):
        
        idx1, idx2 = sorted_idx[i], sorted_idx[i + 1]
        x1, width1 = stats[idx1, cv2.CC_STAT_LEFT], stats[idx1, cv2.CC_STAT_WIDTH]
        x2, width2 = stats[idx2, cv2.CC_STAT_LEFT], stats[idx2, cv2.CC_STAT_WIDTH]
        space_width = x2 - (x1 + width1)
    
        # width size가 20 이상인 contours만 검사
        if width1 >= 50 and width2 >= 50:
            if min_space_th < space_width <= max_space_th:
                cv2.putText(output_check, f"Distance: {space_width}", (x1 + width1, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.rectangle(output_check, (x1 + width1, 0), (x2, img.shape[0]), (0, 0, 255), 1)

    '''찾은 contours draw'''
    for idx in filtered_idx:
        if idx == 0:
            continue
        x, y, width, height, area = stats[idx]
        cv2.rectangle(output_check_temp, (x, y), (x + width, y + height), (0, 255, 0), 1)
        cv2.putText(output_check_temp, f"Area: {area}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # temp = draw_empty_spaces_between_contours(output_check, stats, sorted_idx, min_space_th, max_space_th)
    cv2.imshow('Distance', output_check)
    cv2.imshow('Result', output_check_temp)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < 20:
            continue
        mask = labels == label
        edges = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
 
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
            iStartX, iEndX, lstTopBoundary, lstBotBoundary = calculate_boundaries(contour, img.shape)
            output_check_peak = draw_boundaries(output_check_peak, iStartX, iEndX, lstTopBoundary, lstBotBoundary)
            lstTopBoundary = [img.shape[0]] * img.shape[1]
            lstBotBoundary = [-1] * img.shape[1]
 
            iStartX = img.shape[1]
            iEndX = -1
           
            for pt in contour:
                x, y = pt[0]
                
                lstTopBoundary[x] = min(lstTopBoundary[x], y)
                lstBotBoundary[x] = max(lstBotBoundary[x], y)

                iStartX = min(iStartX, x)
                iEndX = max(iEndX, x)

            # Draw inflection points
            inflection_points_top = find_m_center(lstTopBoundary[iStartX:iEndX], dist_th, height_th)
            inflection_points_bot = find_w_center(lstBotBoundary[iStartX:iEndX], dist_th, height_th)
            
            # inflection_points_top = find_m_center(lstTopBoundary[iStartX:iEndX])
            # inflection_points_bot = find_w_center(lstBotBoundary[iStartX:iEndX])

            if inflection_points_top is not None:
                for x, change in inflection_points_top:  # x가 튜플의 첫 번째 요소임을 분명히 함
                    center_coordinates = (x + iStartX, lstTopBoundary[x + iStartX])
                    cv2.circle(output_check_peak, center_coordinates, 2, (0,0,0), -1)
           
            if inflection_points_bot is not None:
                for x, change in inflection_points_bot:  # 마찬가지로 x는 튜플의 첫 번째 요소
                    center_coordinates = (x + iStartX, lstBotBoundary[x + iStartX])
                    cv2.circle(output_check_peak, center_coordinates, 2, (0,0,255), -1)
    
    # Show results
    cv2.imshow('Output with Inflection Points', output_check_peak)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''process_image(Input Image, Peaks X Distance, Contour Point Heigth Th)'''
if __name__ == '__main__':
    process_image('test.bmp', 50, 3)