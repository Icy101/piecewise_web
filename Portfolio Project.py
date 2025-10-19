car_num = int(input("Select a car number: "))
driver = int(input("Select a driver number: "))
distance = int(input("How long is the race in kilometers?: "))
road_conditions = input("Are the road conditions wet? (True or False): ")

if car_num == 1:
    car_speed = 250
    car_range = 60 * 7.3
elif car_num == 2:
    car_speed = 311
    car_range = 64 * 7.2
elif car_num == 3:
    car_speed = 375
    car_range = 70 * 5.2
else:
    print("Input number not recognized.")
    car_num = 0

if car_num != 0:
    if driver == 1:
        recklessness = 1.1
    elif driver == 2:
        recklessness = 1.2
    elif driver == 3:
        recklessness = 1.3
    else:
        print("Input number not recognized.")
        driver = 0
    
    if driver != 0:
        if road_conditions == "True":
            road_modifier = 1.2
        elif road_conditions == "False":
            road_modifier = 1.0
        else:
            print("Input not recognized.")
            road_modifier = 0
    
        if road_modifier != 0:
            time_spent_driving = distance/car_speed
            if distance > car_range:
                num_refuels = distance//car_range
               
            else:
                num_refuels = 0
                
            total_time = (time_spent_driving + (num_refuels * (5/60)))*recklessness*road_modifier 
            print("The total travel time for Car " + str(car_num) + " with Driver " + str(driver) + " to travel " + str(distance) + " km is " + str(round(total_time, 1)) + " hours.")
                

