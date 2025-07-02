import pygame as pg
import time
import numpy
import copy
from queue import PriorityQueue
from collections import deque
import random
import psutil
import os

MAPSIZE = 6
MAPPOSX = 60
MAPPOSY = 85
WINX = 5
WINY = 2
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60
COLOR = [
    (255,255,255),
    (255,0,0),
    (255,255,0),
    (0,255,255),
    (0,255,0),
    (0,0,255),
    (127,0,0),
    (127,127,0),
    (127,0,127),
    (0,127,127),
    (0,127,0),
    (0,0,127),
    (255, 0, 255)
]
MAIN_MENU = "main_menu"
CHOOSE_MAP = "choose_map"
CHOOSE_ALGORITHM = "choose_algorithm"
ALGORITHM_INFO = "algorithm_info"
SOLUTION = "solution"
BACKGROUND = [
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
]

mainScreen = None
clock = None

class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = 0
        self.peak_memory = 0
        self.current_memory = 0
    
    def start_tracking(self):
        """Start memory tracking"""
        self.start_memory = self.process.memory_info().rss / 1024  # KB
        self.peak_memory = self.start_memory
        self.current_memory = self.start_memory
    
    def update(self):
        """Update current memory usage"""
        self.current_memory = self.process.memory_info().rss / 1024  # KB
        if self.current_memory > self.peak_memory:
            self.peak_memory = self.current_memory
    
    def get_usage(self):
        """Get memory usage statistics"""
        self.update()
        return {
            'current': self.current_memory,
            'peak': self.peak_memory,
            'used': self.current_memory - self.start_memory,
            'peak_used': self.peak_memory - self.start_memory
        }
    
    def get_formatted_usage(self):
        """Get formatted memory usage string"""
        usage = self.get_usage()
        return f"Memory: {usage['used']:.1f}KB (Peak: {usage['peak_used']:.1f}KB)"


class Button:
    def __init__(self, x: float, y: float, width:float, height:float, text:str, font_size: int = 36) -> None:
        self.rect = pg.Rect(x, y, width, height)
        self.color = (0, 0, 0)  # Nền đen bình thường
        self.hover_color = (255, 255, 255)  # Nền trắng khi hover
        self.border_color = (255, 255, 255)  # Viền trắng
        self.border_width = 2
        self.text_color = (255, 255, 255)  # Chữ trắng bình thường
        self.text_hover_color = (0, 0, 0)  # Chữ đen khi hover
        self.clicked = False
        self.is_pressed = False  
        self.last_action_time = 0 
        self.was_pressed_last_frame = False  
        self.text = text
        self.font = pg.font.Font(None, font_size) 
        # Tạo text surface cho cả trạng thái bình thường và hover
        self.text_surface = self.font.render(self.text, True, self.text_color)
        self.text_surface_hover = self.font.render(self.text, True, self.text_hover_color)
        # Căn giữa text trong button
        self.text_rect = self.text_surface.get_rect(center=self.rect.center)

    def draw(self, screen: pg.Surface, enable_hold: bool = False, delay: int = 200) -> bool:
        action = False
        mouse_pos = pg.mouse.get_pos()
        mouse_pressed = pg.mouse.get_pressed()[0]
        now = pg.time.get_ticks()
        
        fresh_click = mouse_pressed and not self.was_pressed_last_frame
        
        if self.rect.collidepoint(mouse_pos):
            # Khi hover: nền trắng + chữ đen
            pg.draw.rect(screen, self.hover_color, self.rect)
            pg.draw.rect(screen, self.border_color, self.rect, self.border_width)
            screen.blit(self.text_surface_hover, self.text_rect)
            
            if mouse_pressed:
                if enable_hold:
                    if now - self.last_action_time > delay:
                        self.last_action_time = now
                        action = True
                else:
                    if fresh_click and now - self.last_action_time > 200:  
                        self.clicked = True
                        self.last_action_time = now
                        action = True
        else:
            # Bình thường: nền đen + viền trắng + chữ trắng
            pg.draw.rect(screen, self.color, self.rect)
            pg.draw.rect(screen, self.border_color, self.rect, self.border_width)
            screen.blit(self.text_surface, self.text_rect)
        
        self.was_pressed_last_frame = mouse_pressed
        
        if not mouse_pressed and not enable_hold:
            self.clicked = False
            
        return action
    
    def is_being_held_with_delay(self, delay: int = 200) -> bool:
        mouse_pos = pg.mouse.get_pos()
        mouse_pressed = pg.mouse.get_pressed()[0]
        now = pg.time.get_ticks()
        
        if mouse_pressed and self.rect.collidepoint(mouse_pos):
            if now - self.last_action_time > delay:
                self.last_action_time = now
                return True
        return False
    
    def is_hovered(self) -> bool:
        mouse_pos = pg.mouse.get_pos()
        return self.rect.collidepoint(mouse_pos)
    
    def reset(self) -> None:
        self.clicked = False
        self.is_pressed = False
        self.last_action_time = 0
        self.was_pressed_last_frame = False
    
class Box:
    def __init__(self, x: float, y: float, width: float, height: float, text: str, font_size: int = 48) -> None:
        self.rect = pg.Rect(x, y, width, height)
        self.color = (0, 0, 0)  # Nền đen bình thường
        self.border_color = (255, 255, 255)  # Viền trắng
        self.border_width = 2
        self.text_color = (255, 255, 255)  # Chữ trắng
        self.text = text
        self.font = pg.font.Font(None, font_size)  # Sử dụng font_size parameter
        self.text_surface = self.font.render(self.text, True, self.text_color)
        self.text_rect = self.text_surface.get_rect(center=self.rect.center)

    def draw(self, screen: pg.Surface) -> None:
        pg.draw.rect(screen, self.color, self.rect)
        pg.draw.rect(screen, self.border_color, self.rect, self.border_width)
        screen.blit(self.text_surface, self.text_rect)

class Map2D:
    def __init__(self, size: int, vehicles: numpy.ndarray) -> None:
        self.map: list[list[int]] = [[0 for _ in range(size)] for _ in range(size)]
        self.vehicles: numpy.ndarray = copy.deepcopy(vehicles)
        for i in range(len(vehicles)):
            if vehicles[i][0] == 0:
                for j in range(vehicles[i][1], vehicles[i][3] + 1):
                    self.map[vehicles[i][2]][j] = i + 1
            else:
                for j in range(vehicles[i][2], vehicles[i][4] + 1):
                    self.map[j][vehicles[i][1]] = i + 1

    
    def display(self, screen: pg.Surface, posx: float, posy: float) -> None:
        pg.draw.rect(screen, (253, 250, 246), pg.Rect(posx, posy, 480, 480))
        pg.draw.rect(screen, (255, 242, 194), pg.Rect(posx, posy + 160, 480, 80))
        first: bool = True
        for i in range(len(self.vehicles)):
            item: list = self.vehicles[i]
            vehiclesSize: int = max(item[4] - item[2], item[3] - item[1]) + 1
            if first:
                pg.draw.rect(screen, (255, 0, 0), pg.Rect(80 * item[1] + posx + 2, 80 * item[2] + posy + 2, (80 * vehiclesSize - 4) * (1 - item[0]) + 76 * item[0], (80 * vehiclesSize - 4) * item[0] + 76 * (1 - item[0])))
                first = False
            else:
                pg.draw.rect(screen, (64, 64, 255), pg.Rect(80 * item[1] + posx + 2, 80 * item[2] + posy + 2, (80 * vehiclesSize - 4) * (1 - item[0]) + 76 * item[0], (80 * vehiclesSize - 4) * item[0] + 76 * (1 - item[0])))

    @staticmethod
    def backtrack(path: list['Vehicles']) -> list['Vehicles']:
        if len(path) == 0: 
            return []
        
        tmp: Vehicles = copy.deepcopy(path[-1])
        res: list[Vehicles] = []
        
        while(tmp.parent != -1):
            res.append(tmp)
            tmp: Vehicles = copy.deepcopy(path[tmp.parent])
        
        res.append(tmp)
        res.reverse()
        # print(f"Backtracked solution with {len(res)} steps")
        return res

class Vehicles:
    def __init__(self, vehiclesList: list[list[int]]) -> None:
        self.list: numpy.ndarray = numpy.array(vehiclesList, dtype= int)
        self.cost: int = 0
        self.parent: int = -1
        self.heuristic: float = 0

    def finalOptimalHeuristic(self) -> float:
        """
        Heuristic tối ưu cuối cùng - OPTIMIZED VERSION
        Tạo map matrix một lần thay vì dùng Map2D object
        """
        # Tạo map matrix hiệu quả hơn
        map_matrix = [[0 for _ in range(MAPSIZE)] for _ in range(MAPSIZE)]
        for i, vehicle in enumerate(self.list):
            if vehicle[0] == 0:  # Horizontal
                for j in range(vehicle[1], vehicle[3] + 1):
                    map_matrix[vehicle[2]][j] = i + 1
            else:  # Vertical
                for j in range(vehicle[2], vehicle[4] + 1):
                    map_matrix[j][vehicle[1]] = i + 1
        
        # 1. Khoảng cách cơ bản đến đích
        distance_to_goal = WINX - self.list[0][3]
        if distance_to_goal <= 0:
            return 0
        
        heuristic_value = distance_to_goal
        
        # 2. Phân tích xe chặn đường
        blocking_cost = 0
        red_row = 2  # Hàng của xe đỏ
        
        for i in range(self.list[0][3] + 1, MAPSIZE):
            if map_matrix[red_row][i] != 0:
                blocking_vehicle_id = map_matrix[red_row][i] - 1
                blocking_vehicle = self.list[blocking_vehicle_id]
                
                if blocking_vehicle[4] - blocking_vehicle[2] == 2:  # Xe dài 3 ô
                    blocking_cost += (MAPSIZE - 1 - blocking_vehicle[4]) * 2.5
                else:
                    blocking_cost += 1.5
                
                # Thêm cascade cost cho xe dọc
                if blocking_vehicle[0] == 1:  # Xe dọc
                    for j in range(blocking_vehicle[2], blocking_vehicle[4] + 1):
                        for k in range(MAPSIZE):
                            if (k != i and map_matrix[j][k] != 0 and 
                                map_matrix[j][k] != blocking_vehicle_id + 1):
                                blocking_cost += 0.5
        
        # 3. Bonus cho không gian trống gần xe đỏ
        space_bonus = 0
        for col in range(self.list[0][3] + 1, min(self.list[0][3] + 3, MAPSIZE)):
            if map_matrix[red_row][col] == 0:
                space_bonus += 0.1
        
        return heuristic_value + blocking_cost - space_bonus

    def moveVehicle(self, index: int, value: bool) -> 'Vehicles':
        '''Di chuyển xe thứ index trong list - OPTIMIZED VERSION\n
        value = 0: lên/qua phải\n
        value = 1: xuống/qua trái\n'''
        # Sử dụng shallow copy thay vì deep copy để tăng tốc
        new_list = self.list.copy()
        isVertical = new_list[index][0]
        if isVertical and value:
            new_list[index][2] += 1
            new_list[index][4] += 1
        elif isVertical:
            new_list[index][2] -= 1
            new_list[index][4] -= 1
        elif not isVertical and value:
            new_list[index][1] -= 1
            new_list[index][3] -= 1
        else:
            new_list[index][1] += 1
            new_list[index][3] += 1
        
        # Tạo object mới
        result = Vehicles(new_list.tolist())
        return result
    
    def validMovements(self) -> list[list[int]]:
        '''Trả về tất cả các nước đi có thể - OPTIMIZED VERSION\n
        Con số thứ nhất là index của xe trong list\n
        Số thứ 2 là value:\n
        0: lên/qua phải, 1: xuống/qua trái'''
        result: list[list[int]] = []
        # Tạo map matrix một lần duy nhất thay vì tạo Map2D object
        map_matrix = [[0 for _ in range(MAPSIZE)] for _ in range(MAPSIZE)]
        for i, vehicle in enumerate(self.list):
            if vehicle[0] == 0:  # Horizontal
                for j in range(vehicle[1], vehicle[3] + 1):
                    map_matrix[vehicle[2]][j] = i + 1
            else:  # Vertical
                for j in range(vehicle[2], vehicle[4] + 1):
                    map_matrix[j][vehicle[1]] = i + 1
        
        # Check valid movements using the created matrix
        for i, vehicle in enumerate(self.list):
            if vehicle[0] == 0:  # Horizontal vehicle
                # Move left
                if vehicle[1] > 0 and map_matrix[vehicle[2]][vehicle[1] - 1] == 0:
                    result.append([i, 1])
                # Move right  
                if vehicle[3] < MAPSIZE - 1 and map_matrix[vehicle[2]][vehicle[3] + 1] == 0:
                    result.append([i, 0])
            else:  # Vertical vehicle
                # Move up
                if vehicle[2] > 0 and map_matrix[vehicle[2] - 1][vehicle[1]] == 0:
                    result.append([i, 0])
                # Move down
                if vehicle[4] < MAPSIZE - 1 and map_matrix[vehicle[4] + 1][vehicle[1]] == 0:
                    result.append([i, 1])
        return result
    
    def winCondition(self) -> bool:
        '''Kiểm tra đã đạt điều kiện để thắng chưa'''
        return self.list[0][3] == WINX and self.list[0][4] == WINY

    def checkVisited(self, listOfVehicles: list['Vehicles'] | deque['Vehicles']) -> bool:
        '''Kiểm tra đã đi qua trường hợp này chưa và thay đổi trường hợp đã đi thành trường hợp có trọng số nhỏ hơn nếu có thể'''
        for i in range(len(listOfVehicles)):
            tmp = listOfVehicles[i]
            if not numpy.array_equal(numpy.array(tmp.list), numpy.array(self.list)):
                continue
            if self < tmp:
                listOfVehicles[i] = self
            return True
        return False
    
    def _checkVisited(self, listOfVehicles: list['Vehicles'] | deque['Vehicles']) -> bool:
        '''Kiểm tra đã đi qua trường hợp này chưa mà không quan tâm đến trọng số'''
        for i in range(len(listOfVehicles)):
            tmp = listOfVehicles[i]
            if not numpy.array_equal(numpy.array(tmp.list), numpy.array(self.list)):
                continue
            return True
        return False
    
    def get_state_hash(self) -> int:
        """Tạo hash cho state để so sánh nhanh - cached version"""
        if not hasattr(self, '_cached_hash'):
            self._cached_hash = hash(tuple(map(tuple, self.list)))
        return self._cached_hash

    def checkVisitedOptimized(self, visited_set: set) -> bool:
        """Kiểm tra visited bằng hash set - O(1) thay vì O(n)"""
        state_hash = self.get_state_hash()
        if state_hash in visited_set:
            return True
        visited_set.add(state_hash)
        return False
    
    def __lt__(self, value: 'Vehicles') -> bool:
        return self.cost + self.heuristic < value.cost + value.heuristic 

class SearchAlgorithm:
    def __init__(self, vehicles: Vehicles):
        self.currentAlgorithm: function = None
        self.startMap: Vehicles = vehicles
        self.show_progress = True
        self.memory_tracker = None
    
    def show_algorithm_progress(self, algorithm_name: str, explored_count: int, frontier_count: int, current_state: Vehicles = None):
        """Hiển thị tiến trình thuật toán real-time - OPTIMIZED VERSION"""
        if not self.show_progress:
            return
            
        # Update memory tracking if available
        if self.memory_tracker:
            self.memory_tracker.update()
            
        # Giảm delay để tăng tốc
        pg.time.delay(10)  # Giảm từ 50ms xuống 10ms
            
        # Xóa màn hình
        mainScreen.fill((0, 0, 0))
        
        # Vẽ title và border
        title_box = Box(0, 0, SCREEN_WIDTH, 50, "RUSH HOUR", 70)
        border_box = Box(0, 50, SCREEN_WIDTH, SCREEN_HEIGHT - 50, " ", 48)
        partition_box = Box(0, 50, 600, SCREEN_HEIGHT - 50, " ", 48)
        
        title_box.draw(mainScreen)
        border_box.draw(mainScreen)
        partition_box.draw(mainScreen)
        
        # Vẽ bản đồ hiện tại
        if current_state:
            Map2D(MAPSIZE, current_state.list).display(mainScreen, MAPPOSX, MAPPOSY)
            # Vẽ viền vàng để hiển thị nó đang được khám phá
            pg.draw.rect(mainScreen, (255, 255, 0), pg.Rect(MAPPOSX - 5, MAPPOSY - 5, 80 * MAPSIZE + 10, 80 * MAPSIZE + 10), 3)

        # Vẽ thông tin thuật toán
        font = pg.font.Font(None, 48)
        text_surface = font.render(f"{algorithm_name}", True, (255, 255, 255))
        mainScreen.blit(text_surface, (620, 100))
        
        stats_font = pg.font.Font(None, 32)
        stats_text = [
            f"Exploring nodes...",
            f"Explored: {explored_count}",
            f"Frontier: {frontier_count}",
            f"Total: {explored_count + frontier_count}"
        ]
        
        # Add memory usage to progress display
        if self.memory_tracker:
            memory_usage = self.memory_tracker.get_usage()
            stats_text.append(f"Memory: {memory_usage['used']:.1f}KB")
        
        if current_state:
            stats_text.append(f"Current cost: {current_state.cost}")
            if hasattr(current_state, 'heuristic'):
                stats_text.append(f"Heuristic: {current_state.heuristic:.1f}")
        
        for i, text in enumerate(stats_text):
            text_surface = stats_font.render(text, True, (255, 255, 255))
            mainScreen.blit(text_surface, (620, 150 + i * 35))
        
        pg.display.update()

    def BFS(self) -> tuple[list[Vehicles], int]:
        f: deque[Vehicles] = deque()
        e: list[Vehicles] = []
        visited_hashes = set()  # Sử dụng hash set thay vì checkVisited
        f.append(self.startMap)
        visited_hashes.add(self.startMap.get_state_hash())
        iteration_count: int = 0
        
        while(len(f) > 0):
            curr: Vehicles = f.popleft()
            e.append(curr)
            
            # Show progress every 20 iterations để giảm overhead
            iteration_count += 1
            if self.show_progress and (iteration_count % 20 == 0 or curr.winCondition()):
                self.show_algorithm_progress("BFS", len(e), len(f), curr)
            
            if curr.winCondition():
                if self.show_progress:
                    self.show_algorithm_progress("BFS", len(e), len(f), curr)
                    pg.time.delay(1000)
                return e, len(f) + len(e)
            
            for i in curr.validMovements():
                tmp: Vehicles = curr.moveVehicle(i[0], i[1])
                tmp.cost = curr.cost + max(tmp.list[i[0]][3] - tmp.list[i[0]][1], tmp.list[i[0]][4] - tmp.list[i[0]][2]) + 1
                tmp.parent = len(e) - 1
                if not tmp.checkVisitedOptimized(visited_hashes):
                    f.append(tmp)
        return []
    
    def DFS(self) -> tuple[list[Vehicles], int]:
        f: deque[Vehicles] = deque()
        e: list[Vehicles] = []
        visited_hashes = set()  # Sử dụng hash set
        f.append(self.startMap)
        visited_hashes.add(self.startMap.get_state_hash())
        iteration_count: int = 0
        
        while(len(f) > 0):
            curr: Vehicles = f.pop()
            e.append(curr)
            
            # Show progress every 20 iterations để giảm overhead
            iteration_count += 1
            if self.show_progress and (iteration_count % 20 == 0 or curr.winCondition()):
                self.show_algorithm_progress("DFS", len(e), len(f), curr)
            
            if curr.winCondition():
                if self.show_progress:
                    self.show_algorithm_progress("DFS", len(e), len(f), curr)
                    pg.time.delay(1000)
                return e, len(f) + len(e)
            
            for i in curr.validMovements():
                tmp: Vehicles = curr.moveVehicle(i[0], i[1])
                tmp.cost = curr.cost + max(tmp.list[i[0]][3] - tmp.list[i[0]][1], tmp.list[i[0]][4] - tmp.list[i[0]][2]) + 1
                tmp.parent = len(e) - 1
                if not tmp.checkVisitedOptimized(visited_hashes):
                    f.append(tmp)
        return []
    


    def AStar(self) -> tuple[list[Vehicles], int]:
        """A* sử dụng priority queue và hash set cho hiệu suất tối ưu"""
        frontier: PriorityQueue = PriorityQueue()
        explored = []
        visited_hashes = set()
        iteration_count = 0
        counter = 0  # Để break ties trong priority queue
        
        # Thêm start state
        self.startMap.heuristic = self.startMap.finalOptimalHeuristic()
        frontier.put((self.startMap.cost + self.startMap.heuristic, counter, self.startMap))
        visited_hashes.add(self.startMap.get_state_hash())
        counter += 1
        
        while not frontier.empty():
            curr: Vehicles
            _, _, curr = frontier.get()
            explored.append(curr)
            
            # Giảm tần suất hiển thị progress để tăng tốc
            iteration_count += 1
            if self.show_progress and (iteration_count % 20 == 0 or curr.winCondition()):
                self.show_algorithm_progress("A*", len(explored), frontier.qsize(), curr)
            
            if curr.winCondition():
                if self.show_progress:
                    self.show_algorithm_progress("A*", len(explored), frontier.qsize(), curr)
                    pg.time.delay(1000)
                return explored, len(frontier) + len(explored)
            
            # Expand current node
            for move in curr.validMovements():
                child = curr.moveVehicle(move[0], move[1])
                child.cost = curr.cost + max(
                    child.list[move[0]][3] - child.list[move[0]][1],
                    child.list[move[0]][4] - child.list[move[0]][2]
                ) + 1
                child.heuristic = child.finalOptimalHeuristic()
                child.parent = len(explored) - 1
                
                # Check if state already visited using hash
                if not child.checkVisitedOptimized(visited_hashes):
                    frontier.put((child.cost + child.heuristic, counter, child))
                    counter += 1
        
        return []

    def AStarOptimized(self) -> tuple[list[Vehicles], int]:
        """A* tối ưu - sử dụng PriorityQueue và hash set cho hiệu suất cao nhất"""
        frontier: PriorityQueue = PriorityQueue()
        explored: list = []
        visited_hashes = set()
        iteration_count = 0
        counter = 0  # Để break ties trong priority queue
        
        # Thêm start state
        self.startMap.heuristic = self.startMap.finalOptimalHeuristic()
        frontier.put((self.startMap.cost + self.startMap.heuristic, counter, self.startMap))
        visited_hashes.add(self.startMap.get_state_hash())
        counter += 1
        
        while not frontier.empty():
            curr: Vehicles
            _, _, curr = frontier.get()
            explored.append(curr)
            
            # Giảm tần suất hiển thị progress để tăng tốc tối đa
            iteration_count += 1
            if self.show_progress and (iteration_count % 50 == 0 or curr.winCondition()):
                self.show_algorithm_progress("A* Optimized", len(explored), frontier.qsize(), curr)
            
            if curr.winCondition():
                if self.show_progress:
                    self.show_algorithm_progress("A* Optimized", len(explored), frontier.qsize(), curr)
                    pg.time.delay(1000)
                return explored, len(explored) + frontier.qsize()
            
            # Expand current node
            for move in curr.validMovements():
                child = curr.moveVehicle(move[0], move[1])
                child.cost = curr.cost + max(
                    child.list[move[0]][3] - child.list[move[0]][1],
                    child.list[move[0]][4] - child.list[move[0]][2]
                ) + 1
                child.heuristic = child.finalOptimalHeuristic()
                child.parent = len(explored) - 1
                
                # Check if state already visited using hash
                if not child.checkVisitedOptimized(visited_hashes):
                    frontier.put((child.cost + child.heuristic, counter, child))
                    counter += 1
        
        return []

def main():
    global mainScreen, clock
    
    # Khởi tạo pygame và màn hình
    pg.init()
    pg.display.set_caption("AI MIDTERM TEST - RUSH HOUR")
    mainScreen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pg.time.Clock()
    
    # Game state variables
    current_state = MAIN_MENU
    current_map_index = 0
    selected_algorithm = None
    algorithm_result = None
    solution_index = 0
    running = True
    state_change_time = 0 
    show_progress = True 
    
    # Map configurations
    map_configs = [
        [[0, 0, 2, 1, 2]],
        [[0, 0, 2, 1, 2], [1, 4, 0, 4, 2], [0, 4, 4, 5, 4]],
        [[0, 0, 2, 1, 2], [1, 5, 0, 5, 2], [0, 4, 3, 5, 3], [1, 2, 4, 2, 5], [0, 3, 5, 5, 5]],
        [[0, 0, 2, 1, 2], [0, 0, 0, 1, 0], [1, 2, 0, 2, 1], [1, 3, 0, 3, 2], [1, 0, 3, 0, 5], [0, 2, 3, 4, 3], [0, 2, 5, 4, 5], [1, 5, 4, 5, 5]],
        [[0, 2, 2, 3, 2], [0, 0, 0, 1, 0], [0, 4, 1, 5, 1], [0, 2, 3, 3, 3], [0, 3, 4, 4, 4], [0, 0, 5, 1, 5], [1, 0, 2, 0, 3], [1, 1, 1, 1, 2], [1, 2, 4, 2, 5], [1, 3, 0, 3, 1], [1, 4, 2, 4, 3], [1, 5, 2, 5, 3]]
    ]
    
    # UI
    title_box = Box(0, 0, SCREEN_WIDTH, 50, "RUSH HOUR", 70)
    border_box = Box(0, 50, SCREEN_WIDTH, SCREEN_HEIGHT - 50, " ", 48)
    partition_box = Box(0, 50, 600, SCREEN_HEIGHT - 50, " ", 48)
    
    # Vẽ UI cơ bản
    main_start_button = Button(620, 150, 360, 50, "START", 32)
    main_exit_button = Button(620, 250, 360, 50, "EXIT", 32)

    # Các nút chọn bản đồ
    map_next_button = Button(620, 150, 360, 50, "NEXT", 32)
    map_prev_button = Button(620, 220, 360, 50, "PREV", 32)
    map_choose_button = Button(620, 290, 360, 50, "CHOOSE", 32)
    map_exit_button = Button(620, 360, 360, 50, "EXIT", 32)

    # Các nút chọn thuật toán
    algo_bfs_button = Button(620, 150, 360, 50, "BFS", 32)
    algo_dfs_button = Button(620, 220, 360, 50, "DFS", 32)
    algo_astar_button = Button(620, 290, 360, 50, "A*", 32)
    
    # Bật/tắt hiển thị tiến trình
    show_progress_button = Button(620, 430, 200, 40, "Show Progress: ON", 28)
    
    # Game loop chính
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        
        # Xóa màn hình
        mainScreen.fill((0, 0, 0))
        
        # Vẽ UI cơ bản
        title_box.draw(mainScreen)
        border_box.draw(mainScreen)
        partition_box.draw(mainScreen)
        
        previous_state = current_state
        now = pg.time.get_ticks()
        
        if now - state_change_time > 300:
            if current_state == MAIN_MENU:
                current_state = handle_main_menu(mainScreen, main_start_button, main_exit_button)
                
            elif current_state == CHOOSE_MAP:
                current_state, current_map_index = handle_choose_map(mainScreen, current_map_index, map_configs, 
                                                                    map_next_button, map_prev_button, map_choose_button, map_exit_button)
                
            elif current_state == CHOOSE_ALGORITHM:
                current_state, selected_algorithm, show_progress = handle_choose_algorithm(mainScreen, algo_bfs_button, 
                                                                           algo_dfs_button, algo_astar_button, show_progress_button, show_progress)
                
            elif current_state == ALGORITHM_INFO:
                current_state, algorithm_result = handle_algorithm_info(mainScreen, selected_algorithm, map_configs[current_map_index], show_progress)
                
            elif current_state == SOLUTION:
                current_state, solution_index = handle_solution(mainScreen, algorithm_result, solution_index)
        else:

            if current_state == MAIN_MENU:
                main_start_button.draw(mainScreen)
                main_exit_button.draw(mainScreen)
                
            elif current_state == CHOOSE_MAP:
                font = pg.font.Font(None, 36)
                text_surface = font.render("CHOOSE MAP", True, (255, 255, 255))
                mainScreen.blit(text_surface, (620, 100))
                map_next_button.draw(mainScreen, enable_hold=True, delay=300)
                map_prev_button.draw(mainScreen, enable_hold=True, delay=300)
                map_choose_button.draw(mainScreen)
                map_exit_button.draw(mainScreen)
                
            elif current_state == CHOOSE_ALGORITHM:
                font = pg.font.Font(None, 36)
                text_surface = font.render("CHOOSE ALGORITHM", True, (255, 255, 255))
                mainScreen.blit(text_surface, (620, 100))
                algo_bfs_button.draw(mainScreen)
                algo_dfs_button.draw(mainScreen)
                algo_astar_button.draw(mainScreen)
                show_progress_button.draw(mainScreen)
        
        if previous_state != current_state:
            state_change_time = now  
            # Reset main menu buttons
            main_start_button.reset()
            main_exit_button.reset()
            
            # Reset map buttons
            map_next_button.reset()
            map_prev_button.reset()
            map_choose_button.reset()
            map_exit_button.reset()
            
            # Reset algorithm buttons
            algo_bfs_button.reset()
            algo_dfs_button.reset()
            algo_astar_button.reset()
            show_progress_button.reset()
        
        # Vẽ map
        if current_state != MAIN_MENU:
            Map2D(MAPSIZE, map_configs[current_map_index]).display(mainScreen, MAPPOSX, MAPPOSY)    
        pg.display.update()
        clock.tick(FPS)
    
    pg.quit()

def handle_main_menu(screen, start_button, exit_button):
    """Xử lý main menu state"""
    font = pg.font.Font(None, 36)
    text_surface = font.render("MAIN MENU", True, (255, 255, 255))
    screen.blit(text_surface, (620, 100))
    
    empty_map = Map2D(MAPSIZE, [])
    empty_map.display(screen, MAPPOSX, MAPPOSY)
    
    if start_button.draw(screen):
        return CHOOSE_MAP
    if exit_button.draw(screen):
        pg.quit()
        exit()
    
    return MAIN_MENU

def handle_choose_map(screen, map_index, map_configs, next_button, prev_button, choose_button, exit_button):
    """Xử lý choose map state"""
    font = pg.font.Font(None, 36)
    text_surface = font.render("CHOOSE MAP", True, (255, 255, 255))
    screen.blit(text_surface, (620, 100))
    
    new_state = CHOOSE_MAP
    new_index = map_index
    
    if next_button.draw(screen, enable_hold=True, delay=300):
        new_index = (map_index + 1) % len(map_configs)
    if prev_button.draw(screen, enable_hold=True, delay=300):
        new_index = (map_index - 1) % len(map_configs)
    if choose_button.draw(screen):
        new_state = CHOOSE_ALGORITHM
    if exit_button.draw(screen):
        new_state = MAIN_MENU
    
    return new_state, new_index

def handle_choose_algorithm(screen, bfs_button, dfs_button, astar_button, show_progress_button, show_progress):
    """Xử lý choose algorithm state"""
    font = pg.font.Font(None, 36)
    text_surface = font.render("CHOOSE ALGORITHM", True, (255, 255, 255))
    screen.blit(text_surface, (620, 100))
    
    # Update show progress button text
    show_progress_button.text = f"Show Progress: {'ON' if show_progress else 'OFF'}"
    show_progress_button.text_surface = show_progress_button.font.render(show_progress_button.text, True, show_progress_button.text_color)
    show_progress_button.text_surface_hover = show_progress_button.font.render(show_progress_button.text, True, show_progress_button.text_hover_color)
    
    new_show_progress = show_progress
    
    if bfs_button.draw(screen):
        return ALGORITHM_INFO, "BFS", new_show_progress
    if dfs_button.draw(screen):
        return ALGORITHM_INFO, "DFS", new_show_progress
    if astar_button.draw(screen):
        return ALGORITHM_INFO, "A*", new_show_progress
    if show_progress_button.draw(screen):
        new_show_progress = not show_progress
    
    return CHOOSE_ALGORITHM, None, new_show_progress

def handle_algorithm_info(screen, algorithm, map_config, show_progress):
    """Xử lý algorithm info state và chạy algorithm"""
    font = pg.font.Font(None, 48)
    text_surface = font.render(f"{algorithm}", True, (255, 255, 255))
    screen.blit(text_surface, (620, 100))
    
    # Show "Running..." message
    running_font = pg.font.Font(None, 32)
    running_text = running_font.render("Running...", True, (255, 255, 0))
    screen.blit(running_text, (620, 150))
    
    pg.display.update()
    
    # Initialize memory tracking
    memory_tracker = MemoryTracker()
    memory_tracker.start_tracking()
    
    # Chạy thuật toán
    starting_map = Vehicles(map_config)
    search_algo = SearchAlgorithm(starting_map)
    search_algo.show_progress = show_progress  # Set the show_progress flag
    search_algo.memory_tracker = memory_tracker  # Pass memory tracker to search algorithm
    
    # Hiển thị thông tin thuật toán
    start_time = pg.time.get_ticks()
    
    if algorithm == "BFS":
        result, total_nodes = search_algo.BFS()
    elif algorithm == "DFS":
        result, total_nodes = search_algo.DFS()
    elif algorithm == "A*":
        result, total_nodes = search_algo.AStarOptimized() 
    else:
        result = []
    
    end_time = pg.time.get_ticks()
    execution_time = end_time - start_time
    
    # Get final memory statistics
    memory_stats = memory_tracker.get_usage()
    
    # Xóa màn hình và hiện thị kết quả
    mainScreen.fill((0, 0, 0))
    
    # Draw base UI
    title_box = Box(0, 0, SCREEN_WIDTH, 50, "RUSH HOUR", 70)
    border_box = Box(0, 50, SCREEN_WIDTH, SCREEN_HEIGHT - 50, " ", 48)
    partition_box = Box(0, 50, 600, SCREEN_HEIGHT - 50, " ", 48)
    
    title_box.draw(mainScreen)
    border_box.draw(mainScreen)
    partition_box.draw(mainScreen)
    
    # Vẽ map hiện tại
    Map2D(MAPSIZE, map_config).display(mainScreen, MAPPOSX, MAPPOSY)
    # Vẽ viền xanh để hiển thị nó đang được khám phá
    pg.draw.rect(mainScreen, (0, 255, 0), pg.Rect(MAPPOSX - 5, MAPPOSY - 5, 80 * MAPSIZE + 10, 80 * MAPSIZE + 10), 5)

    # Vẽ kết quả thuật toán
    result_font = pg.font.Font(None, 48)
    result_text = result_font.render(f"{algorithm}", True, (255, 255, 255))
    screen.blit(result_text, (620, 100))
    
    stats_font = pg.font.Font(None, 32)
    
    # Tính toán và hiển thị thống kê
    if result:
        # total_nodes = len(result)
        final_cost = result[-1].cost if result else 0
        final_heuristic = result[-1].heuristic if result else 0
        
        stats_text = [
            f"Total nodes: {total_nodes}",
            f"Cost: {final_cost}",
            f"Heuristic: {final_heuristic:.1f}",
            f"Time: {execution_time}ms",
            f"Memory: {memory_stats['used']:.1f}KB",
            f"Peak Memory: {memory_stats['peak_used']:.1f}KB",
            f"Status: Solution found!"
        ]
        text_color = (0, 255, 0)  # Green for success
    else:
        stats_text = [
            "Total nodes: 0",
            "Cost: N/A",
            "Heuristic: N/A",
            f"Time: {execution_time}ms",
            f"Memory: {memory_stats['used']:.1f}KB",
            f"Peak Memory: {memory_stats['peak_used']:.1f}KB",
            "Status: No solution found!"
        ]
        text_color = (255, 0, 0)  # Red for failure
    
    for i, text in enumerate(stats_text):
        if i == len(stats_text) - 1:  # Status line
            text_surface = stats_font.render(text, True, text_color)
        else:
            text_surface = stats_font.render(text, True, (255, 255, 255))
        screen.blit(text_surface, (620, 150 + i * 35))
    
    # Thêm nút CONTINUE
    continue_button = Button(620, 350, 360, 50, "CONTINUE", 32)
    continue_clicked = False
    
    # Đợi người dùng nhấn CONTINUE
    waiting = True
    while waiting:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                exit()
        
        if continue_button.draw(screen):
            waiting = False
        
        pg.display.update()
        clock.tick(FPS)
    
    return SOLUTION, result

def show_steps(res, back_button, next_button, out_button):
    """Hiển thị từng bước của solution với navigation"""
    idx = 0
    running = True
    delay = 150  # milliseconds
    last_action_time = 0
    user_wants_exit = False
    
    # Reset buttons khi bắt đầu
    back_button.reset()
    next_button.reset()
    out_button.reset()
    
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return "quit_program"
        
        mainScreen.fill((0, 0, 0))
        
        title_box = Box(0, 0, SCREEN_WIDTH, 50, "RUSH HOUR", 70)
        border_box = Box(0, 50, SCREEN_WIDTH, SCREEN_HEIGHT - 50, " ", 48)
        partition_box = Box(0, 50, 600, SCREEN_HEIGHT - 50, " ", 48)
        
        title_box.draw(mainScreen)
        border_box.draw(mainScreen)
        partition_box.draw(mainScreen)
        
        # Hiển thị bản đồ hiện tại
        if res and idx < len(res):
            Map2D(MAPSIZE, res[idx].list).display(mainScreen, MAPPOSX, MAPPOSY)
            # Vẽ viền xanh để hiển thị nó đang được khám phá
            pg.draw.rect(mainScreen, (0, 255, 0), pg.Rect(MAPPOSX - 5, MAPPOSY - 5, 80 * MAPSIZE + 10, 80 * MAPSIZE + 10), 5)

            # Hiển thị thông tin bước
            font = pg.font.Font(None, 36)
            step_text = font.render(f"Step: {idx + 1}/{len(res)}", True, (255, 255, 255))
            mainScreen.blit(step_text, (620, 100))
            
            cost_text = font.render(f"Cost: {res[idx].cost}", True, (255, 255, 255))
            mainScreen.blit(cost_text, (620, 140))
        
        # Vẽ các nút điều hướng
        back_clicked = back_button.draw(mainScreen, enable_hold=True, delay=delay)
        next_clicked = next_button.draw(mainScreen, enable_hold=True, delay=delay)
        out_clicked = out_button.draw(mainScreen)
        
        # Xử lý các nút điều hướng
        if back_clicked and idx > 0:
            idx -= 1
        if next_clicked and idx < len(res) - 1:
            idx += 1
        if out_clicked:
            # Chờ cho đến khi chuột được thả để tránh bị trùng click
            while pg.mouse.get_pressed()[0]:
                pg.time.delay(10)
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        return "quit_program"
            pg.time.delay(100)  # Thêm delay sau khi thả chuột
            return "exit_to_menu"
        
        pg.display.update()
        clock.tick(FPS)

def handle_solution(screen, result, solution_index):
    """Xử lý solution state và backtrack solution"""
    if result:
        # Hiển thị solution
        solution_steps = Map2D.backtrack(result)
        
        # Create buttons for show_steps
        back_button = Button(620, 200, 175, 50, "BACK", 32)
        next_button = Button(805, 200, 175, 50, "NEXT", 32)
        out_button = Button(620, 270, 360, 50, "EXIT", 32)

        # Cho người dùng xem từng bước của solution
        result = show_steps(solution_steps, back_button, next_button, out_button)
        
        if result == "quit_program":
            # Người dùng đóng cửa sổ
            pg.quit()
            exit()
        elif result == "exit_to_menu":
            # Người dùng nhấn nút EXIT, quay lại menu chính
            pg.time.delay(200)  # Thêm delay để tránh click accidental
            return MAIN_MENU, 0
        
        return MAIN_MENU, 0
    else:
        # No solution found
        font = pg.font.Font(None, 36)
        text_surface = font.render("No solution found!", True, (255, 0, 0))
        screen.blit(text_surface, (620, 100))
        
        exit_button = Button(620, 200, 100, 50, "EXIT", 32)
        if exit_button.draw(screen):
            return MAIN_MENU, 0
        
        return SOLUTION, solution_index
if __name__ == "__main__":
    main()