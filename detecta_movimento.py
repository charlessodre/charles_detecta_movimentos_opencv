import cv2
import time

print(cv2.__version__)
# 4.1.1

video_entrada = "./videos/video_movimento.mp4"

# Acessa o video que será analisado.
captura = cv2.VideoCapture(video_entrada)
_, frame_inicial = captura.read()

# Define as configurações do vídeo de saida que vai conter os pontos detectados e movimentos detectados.
video_saida = "./output/saida_video.mp4"
gravar_video = cv2.VideoWriter(video_saida, cv2.VideoWriter_fourcc(*'mp4v'), 10,
                               (frame_inicial.shape[1], frame_inicial.shape[0]))

# Captura pela Camera
# captura = cv2.VideoCapture(0)

total_left_entries = 0
total_right_entries = 0

# Area mínima do contorno considerado.
min_contour_Area = 7000
min_center_distance = 5
gaussian_blur_value = 3
threshold_binary_value = 9

x_current_position = None
x_last_position = None
movement_direction = 0
detected_movement = False

while True:

    ret1, frame1 = captura.read()
    ret2, frame2 = captura.read()

    if not ret1 or not ret2:
        break

    time.sleep(0.03)

    frame_width = frame1.shape[1]
    frame_height = frame1.shape[0]

    # Converte os frames para cinza.
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Aplica o GaussianBlur- Smoothing Images
    frame1_gray_blur = cv2.GaussianBlur(frame1_gray, (gaussian_blur_value, gaussian_blur_value), 0)
    frame2_gray_blur = cv2.GaussianBlur(frame2_gray, (gaussian_blur_value, gaussian_blur_value), 0)

    # Obtêm as diferenças entre os frames.
    frame_diff = cv2.absdiff(frame1_gray_blur, frame2_gray_blur)

    #
    # Aplica o "Simple Thresholding" para tentar destacar a imagem principal (sem o fundo).
    _, frame1_gray_blur_binary = cv2.threshold(frame_diff, threshold_binary_value, 255, cv2.THRESH_BINARY)

    # Aumenta a área da imagem após a aplicação do Image Thresholding.
    frame_dilate = cv2.dilate(frame1_gray_blur_binary, None, iterations=2)

    # Extrai o contorno da imagem analisada.
    contours, _ = cv2.findContours(frame_dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenha o contorno na imagem
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2) # thickness=cv2.FILLED)

    # Desenha a linha central.
    x_linha_central = int(frame_width / 2.1)
    cv2.line(frame1, (x_linha_central, 0), (x_linha_central, frame_height), (255, 0, 0), 4)

    if len(contours) > 0:

        # Obtêm o contorno com a maior área.
        max_cnt = max(contours, key=cv2.contourArea)

        # Obtêm a área do contorno.
        areaContorno = cv2.contourArea(max_cnt)

        if areaContorno > min_contour_Area:

            (x, y, w, h) = cv2.boundingRect(max_cnt)

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (85, 85, 255), 2)

            x_center = int((x + x + w) / 2)
            y_center = int((y + y + h) / 2)
            center_point = (x_center, y_center)
            cv2.circle(frame1, center_point, 10, (255, 80, 170), thickness=-1)

            x_current_position = x_center
            abs_distance_centert = abs(x_current_position - x_linha_central)

            if x_last_position is None:
                x_last_position = x_current_position

            if x_current_position >= x_linha_central and x_current_position <= x_last_position:
                movement_direction -= 1
            elif x_current_position <= x_linha_central and x_current_position >= x_last_position:
                movement_direction += 1

            if abs_distance_centert <= min_center_distance and not detected_movement:
                if movement_direction < -1:
                    total_left_entries += 1
                    detected_movement = True

            if abs_distance_centert <= min_center_distance and not detected_movement:
                if movement_direction > 1:
                    total_right_entries += 1
                    detected_movement = True

            x_last_position = x_current_position

    else:
        movement_direction = 0
        x_current_position = 0
        x_last_position = 0
        detected_movement = False

    cv2.putText(frame1, "Entradas Direita: {}".format(str(total_left_entries)), (480, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame1, "Entradas Esquerda: {}".format(str(total_right_entries)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 255, 0), 1, cv2.LINE_AA)

    if movement_direction < -1:
        cv2.putText(frame1, "Entrada pela direita detectada!".format(str(movement_direction)), (360, frame_height - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
    elif movement_direction > 1:
        cv2.putText(frame1, "Entrada pela esquerda detectada!".format(str(movement_direction)), (10, frame_height - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

    # Exibe a saida Video
    # cv2.imshow('frame_gray - 1', frame1_gray)
    # cv2.imshow('frame_gray_blur - 2', frame1_gray_blur)
    # cv2.imshow('frame_gray_blur_binary - 3', frame1_gray_blur_binary)
    # cv2.imshow('frame_diff - 4', frame_diff)
    cv2.imshow('frame_dilate - 5', frame_dilate)
    cv2.imshow('original', frame1)

    # Salva uma cópia do frame
    gravar_video.write(frame1)

    # Aperte a tecla 'q' para sair.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos.
gravar_video.release()
cv2.destroyAllWindows()
