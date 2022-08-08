0. Загрузить данные и модель предиктора DLIB 

wget https://drive.google.com/file/d/0B8okgV6zu3CCWlU3b3p4bmJSVUU/view?usp=sharing <br>
tar -xzf landmarks_task.tgz <br>
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 <br>
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 <br>

1. Установить окружение из requirements.txt. Я не смог создать человеческий файл из Каггла, надеюсь, ничего не забыл
2. Тренировка модели - train.py. Все параметры выставлены по умолчанию. Для запуска с конструироваинем датасетов с нуля, надо выставить флаг load_dataset_from_file равным нулю. пример запуска:
python train.py --train_data landmarks_task/300W/train landmarks_task/Menpo/train --test_data landmarks_task/300W/test landmarks_task/Menpo/test --load_dataset_from_file 0. Лучшая модель будет сохранена под именем "best_checkpoint.pt"
3. Получение результатов - detect.py. Параметры по умолчанию для Menpo. Чтобы заменить датасет, надо поменять Menpo на 300W.
4. Получение результатов DLIB - detect_dlib.py. Параметры по умолчанию для Menpo. Чтобы заменить датасет, надо поменять Menpo на 300W. Там еще генерится файл с площадью окна, который потом используется для построения CED
5. Получение метрик CED - я слегка изуродовал оригинальный скрипт, добавив параметры predictions_path1 и normalization_path. Первый - путь к данным предсказаний второй модели (отличной от DLIB), а normalization_path - файл с площадью окна.  
