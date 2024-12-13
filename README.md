# Bonus_task

ВНИМАНИЕ
Для параллельной записи параметров в списки для построения графиков и для интегрирования вектора неизвестных параметров был дополнен стандартный класс симулятора.

Для запуска файла adaptive_control.py необходимо:
1) В файле simulator/_simulator.py на 401 строке записать массив из 10 нулевых символов
2) В файле universal_robots_ur5e/scene.xml указать во второй строке путь до файла ur5e_with_mass.xml (include file="ur5e_with_mass.xml"/)

Для запуска файла adaptive_with_damping.py необходимо:
1) В файле simulator/_simulator.py на 401 строке записать массив из 16 нулевых символов
2) В файле universal_robots_ur5e/scene.xml указать во второй строке путь до файла ur5e_with_damping.xml (include file="ur5e_with_damping.xml"/)
