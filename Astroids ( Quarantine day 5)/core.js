    const FPS = 30;
    const SHIP_SIZE = 50;
    const ROTATION_SPEED = 180;
    const SHIP_THURST = 5;
    const SLOW_DOWW = 0.9;               
    const SHIP_EXPLODE_TIME = 0.5;
    const NO_OF_ASTROIDS = 10;
    const ASTROID_SPEED = 3;
    const ASTROID_SIZE = 100;
    const ASTROID_VERTEX = 10;
    const ASTROID_VERTEX_OFFSET = 0.420;
    const SHOW_BOUNDING_BOX = false;
    const SHIP_IMMORTALITY = 6;
    const SHIP_FLASH_TIME = 0.1;
    const LAZER_MAX = 10;
    const LAZER_SPEED = 50 ;
    const LAZER_MAX_DISTANCE = 0.5;
    const LIVES = 3;


/** @type {HTMLCanvasElement} */
var canvas = document.getElementById("mainCanvas");
var ctx = canvas.getContext("2d");
var img = document.getElementById("background");


//set sound variables
var fxThruster = new Audio("sounds/thrust.m4a");
fxThruster.volume = 0.1;
fxThruster.loop = true;

var fxExplode = new Audio("sounds/explode.m4a");
fxExplode.volume = 0.1;

var fxHit = new Audio("sounds/hit.m4a");
fxExplode.volume = 0.5;

var fxMusic = new Audio("sounds/OriginofEscapeCover.wav");


// game parameters
var level, astroids, ship,txt, text_alpha, lives_left , score , highScore =0;
newGame();


function newGame(){
fxMusic.autoplay = true;
fxMusic.volume = 0.69;
fxMusic.play();
fxMusic.loop = true;
window.alert("Welcome to Astroids. \nRules : \n1.Avoid astroids \n2.After every 1500 points you get a new life ");
score  =0;
level = 0;
lives_left = LIVES;
//SHIP SETUP 
ship = newShip();
newLevel();
}

function newLevel(){
txt = "Level " + (level +1);   
text_alpha = 1;
createAstroids();
}    


function newShip()  {
    return  {   x: canvas.width / 2,
                y: canvas.height / 2,
                radius: SHIP_SIZE / 2,
                angle : 90 / 180 * Math.PI ,   // comverting angle into radians
                rot : 0,
                exploding_time: 0.1,
                blinktime: Math.ceil(SHIP_FLASH_TIME * FPS),
                blinkNumber:0.5* Math.ceil(SHIP_IMMORTALITY / SHIP_FLASH_TIME),
                thursting: false, 
                canShoot: true,
                lazer: [],
                thrust: {
                    x: 0,
                    y: 0
                }
    }
}    

function createAstroids() {
    astroids = [];
    var x,y;
    for (let index = 0; index < NO_OF_ASTROIDS + level; index++) {
        do {
            x = Math.floor(Math.random() * canvas.width);
            y = Math.floor(Math.random() * canvas.height);
        }while( DistanceBetween(ship.x , ship.y, x, y) <  ASTROID_SIZE*2 + ship.radius );
        astroids.push(newAstroid(x,y ,Math.ceil(ASTROID_SIZE /2) ));
    }
}   

function DistanceBetween(a,b,c,d){
    return Math.sqrt( Math.pow( a-c, 2) + Math.pow( b-d, 2) );
}

function newAstroid(x,y,r){
    var lvlMulti = 1 + 0.1 * level;
    var roid = {
    x: x ,
    y: y ,
    x_velocity: lvlMulti* Math.random() +  ASTROID_SPEED / FPS * (Math.random() < 0 ? 1 : -1),
    y_velocity: lvlMulti* Math.random() +  ASTROID_SPEED / FPS * (Math.random() < 0 ? 1 : -1),
    ast_r: r,
    ast_a: Math.random() * 2 * Math.PI,
    vertex: Math.floor(Math.random() * (ASTROID_VERTEX + 1) + (ASTROID_VERTEX / 2) ),
    offset: []
    }   
    for (let index = 0; index < roid.vertex; index++) {
        let t = Math.random() * ASTROID_VERTEX_OFFSET*2 + 1 - ASTROID_VERTEX_OFFSET;
        roid.offset.push( t==0 ? 1 : t );
    }
    return roid;
}

function destroyAstroid(index){
    if( astroids[index].ast_r == Math.ceil(ASTROID_SIZE /2) ){
        score += 10;
        astroids.push( newAstroid(astroids[index].x, astroids[index].y, Math.ceil(ASTROID_SIZE /4)  ));
        astroids.push( newAstroid(astroids[index].x, astroids[index].y, Math.ceil(ASTROID_SIZE /4)  ));
    }
    astroids.splice(index,1);
    score += 10;
    if(score %1500 == 0)
        lives_left++;
    if(score > highScore)
        highScore = score;
    if(astroids.length == 0){
        level++;
        newLevel();
    }
}


// Event Handler
document.addEventListener("keydown" , keyDown);
document.addEventListener("keyup", keyUp);

function keyDown(/** @type {KeyboardEvent} */ ev){
    switch(ev.keyCode){
        case 32:            //space bar
            shootLazer();
            break;
        case 65 :           //left arrow
            ship.rot = ROTATION_SPEED / 180 * Math.PI / FPS;
            break;
        case 87 :           // up arrow
            ship.thursting = true; 
            fxThruster.play();       
            break;
        case 68 :           // right arrow
        ship.rot = -ROTATION_SPEED / 180 * Math.PI / FPS;
            break;
        }
}        

function keyUp(/** @type {KeyboardEvent} */ ev){
    switch(ev.keyCode){
        case 32:            //space bar
            ship.canShoot = true;
            break;
        case 65 :           //left arrow
            ship.rot = 0;
            break;
        case 87 :           // up arrow
            ship.thursting = false;
            fxThruster.pause();
            break;
        case 68 :           // right arrow
        ship.rot = 0;
            break;
    }
}

function kabooom(){
    lives_left--;
    ship.exploding_time = 5*Math.ceil(SHIP_EXPLODE_TIME*FPS);
    fxExplode.play();
}

function shootLazer(){
// create lazer 
    if(ship.canShoot && ship.lazer.length < LAZER_MAX){
        ship.lazer.push({
            x:  ship.x +  ship.radius * Math.cos(ship.angle) ,
            y:  ship.y -  ship.radius * Math.sin(ship.angle) ,
            xv: LAZER_SPEED * Math.cos(ship.angle) ,
            yv: -LAZER_SPEED * Math.sin(ship.angle) ,
            dist: 0 ,
        });  
    }    
    // prevent further shoooting
    ship.canShoot = false;
}

function drawLifeCounter(i){
    ctx.fillStyle = "lime";
    ctx.strokeStyle = "lime";
    ctx.beginPath();
    ctx.arc( (canvas.width - 50) + 30*-1*i ,ship.radius, 10, 0, 2*Math.PI, false );
    ctx.fill();
    ctx.stroke();
}



function make_base(){
    base_image = new Image();
    base_image.src = 'background.jpg';
    ctx.drawImage(base_image, 0, 0);
}


//game loop
setInterval( update , 1000/FPS);

function update(){  
    var exploding = ship.exploding_time > 0.1;
    var immortalON = ship.blinkNumber % 2 == 0 ;

    //draw background
    make_base();
    //    ctx.fillStyle = "black";
    //    ctx.fillRect(0, 0, canvas.width, canvas.height);


    // score board 
    ctx.fillStyle = "white";
    ctx.font = "30px courier new";
    ctx.fillText("Score: " +score , ship.radius,ship.radius +20);

        //  high score board 
        ctx.fillStyle = "white";
        ctx.font = "40px courier new";
        ctx.fillText(highScore , canvas.width/2,ship.radius +20);


    //ship movement
        // ship thrust 
            if(ship.thursting){
            ship.thrust.x += SHIP_THURST * Math.cos(ship.angle) / FPS;
            ship.thrust.y -= SHIP_THURST * Math.sin(ship.angle) / FPS;
                // back booster or thruster
                if(!exploding){
                    ctx.fillStyle = "red";
                    ctx.lineWidth = SHIP_SIZE /15;
                    ctx.strokeStyle = "yellow";
                    ctx.beginPath();
                    ctx.moveTo(
                        ship.x - ship.radius * (Math.cos(ship.angle) +  1/2 * Math.sin(ship.angle)),
                        ship.y + ship.radius * ( Math.sin(ship.angle)  -  1/2 * Math.cos(ship.angle) )
                    );
                    ctx.lineTo( // rear centre (behind the ship)
                        ship.x - ship.radius * 6 / 3 * Math.cos(ship.angle) ,
                        ship.y + ship.radius * 6 / 3 * Math.sin(ship.angle) 
                        );
                    ctx.lineTo( // rear right
                        ship.x - ship.radius * ( Math.cos(ship.angle) - 1/2 * Math.sin(ship.angle)),
                        ship.y + ship.radius * ( Math.sin(ship.angle) + 1/2 * Math.cos(ship.angle))
                    );
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke()
                    }
            }  else {                 //apply friciton
                    ship.thrust.x -= SLOW_DOWW * ship.thrust.x / FPS ;
                    ship.thrust.y -= SLOW_DOWW * ship.thrust.y / FPS;
                }
            

    //draw ship
        if(!exploding){
            if(immortalON){
                ctx.strokeStyle = "#FFD700";
                ctx.fillStyle = "#1E90FF";
                ctx.lineWidth = SHIP_SIZE / 20;
                ctx.beginPath();
                ctx.moveTo( // nose of the ship
                    ship.x +  ship.radius * Math.cos(ship.angle),
                    ship.y -  ship.radius * Math.sin(ship.angle)
                );
                ctx.lineTo( // rear left
                    ship.x - ship.radius * ( Math.cos(ship.angle) + Math.sin(ship.angle)),
                    ship.y + ship.radius * ( Math.sin(ship.angle) - Math.cos(ship.angle))
                );
                ctx.lineTo( // rear right
                    ship.x - ship.radius * ( Math.cos(ship.angle) - Math.sin(ship.angle)),
                    ship.y + ship.radius * ( Math.sin(ship.angle) + Math.cos(ship.angle))
                );
                ctx.closePath();
                ctx.stroke()
                ctx.fill();
                }
            if(ship.blinkNumber >0){
                ship.blinktime--;
                if(ship.blinktime == 0){
                    ship.blinktime = Math.ceil(SHIP_FLASH_TIME * FPS);
                    ship.blinkNumber--;
                }
                
            }
        } else {
            ctx.fillStyle = "red";
            ctx.beginPath();
            ctx.arc(ship.x, ship.y, ship.radius * 1.4, 0, Math.PI * 2, false);
            ctx.fill();
            ctx.fillStyle = "orange";
            ctx.beginPath();
            ctx.arc(ship.x, ship.y, ship.radius * 1.1, 0, Math.PI * 2, false);
            ctx.fill();
            ctx.fillStyle = "yellow";
            ctx.beginPath();
            ctx.arc(ship.x, ship.y, ship.radius * 0.8, 0, Math.PI * 2, false);
            ctx.fill();
            

        }
        
    //bounding box for ship
    if(SHOW_BOUNDING_BOX){
        ctx.strokeStyle = "green";
        ctx.beginPath();
        ctx.arc(ship.x, ship.y, ship.radius  * 1.25, 0, Math.PI*2);
        ctx.stroke();
    }   

    //draw astroids

    let x,y,r,a,vert_off;
    for (let i = 0; i < astroids.length; i++) {
    ctx.strokeStyle = "black";
    ctx.fillStyle = "#778899";
    ctx.lineWidth = SHIP_SIZE/20;
    x = astroids[i].x;
    y = astroids[i].y;
    r = astroids[i].ast_r;
    a = astroids[i].ast_a;   
    vertex = astroids[i].vertex;
    vert_off = astroids[i].offset;
        //draw path
            ctx.beginPath();
            ctx.moveTo(
                x + r * vert_off[0] * Math.cos(a),
                y + r * vert_off[0] * Math.sin(a)
            );
        //draw polygon
            for (let j = 1; j < vertex; j++) {
                ctx.lineTo( 
                    x + r * vert_off[j] * Math.cos(a + j * Math.PI*2 / vertex ),
                    y + r * vert_off[j] * Math.sin(a + j * Math.PI*2 / vertex ),
                );
            }
            ctx.closePath();
            ctx.stroke();
            ctx.fill();
        // draw bounding box
        if(SHOW_BOUNDING_BOX){
        ctx.strokeStyle = "blue";
        ctx.beginPath();
        ctx.arc(astroids[i].x, astroids[i].y, astroids[i].ast_r, 0, Math.PI*2);
        ctx.stroke();
        
        }       
    }

    // drawing lazers 
    for (let i = 0; i < ship.lazer.length; i++) {
        ctx.strokeStyle = "red";
        ctx.fillStyle = "red";
        ctx.beginPath();
        ctx.arc(ship.lazer[i].x , ship.lazer[i].y, SHIP_SIZE/20, 0, Math.PI*2);
        ctx.fill();
        ctx.stroke();
        
    }

    for (let i = 0; i < ship.lazer.length; i++) {

        //distance check
        if(ship.lazer[i].dist > LAZER_MAX_DISTANCE * canvas.width){
            ship.lazer.splice(i,1);
            continue;
        }


        ship.lazer[i].x += ship.lazer[i].xv;
        ship.lazer[i].y += ship.lazer[i].yv;

        // calculate disatnce travelled by eeach laser
        ship.lazer[i].dist += Math.sqrt( Math.pow(ship.lazer[i].xv, 2) + Math.pow(ship.lazer[i].yv,2) ); 

        // loop lazers 
        if (ship.lazer[i].x < 0) {
            ship.lazer[i].x = canvas.width;
        } else if (ship.lazer[i].x > canvas.width) {
            ship.lazer[i].x = 0;
        }
        if (ship.lazer[i].y < 0) {
            ship.lazer[i].y = canvas.height;
        } else if (ship.lazer[i].y > canvas.height) {
            ship.lazer[i].y = 0;
        }
    }


    // draw text
    if(text_alpha >= 0){
        ctx.fillStyle = "rgba(255, 255, 255, " + text_alpha + ")";
        ctx.font = "bold 40px courier new ";
        ctx.fillText(txt, canvas.width /2 + 50 , canvas.height * 3/4 - 40 );
        text_alpha -= (0.5  /FPS); 
    }    

    // draw live counter
    for (let i = 0; i < lives_left; i++) {
            drawLifeCounter(i);
    }

    // detect lazer hits
    for (let i =0; i< astroids.length ; i++) {
        for (let j=0;  j < ship.lazer.length  ; j++) {
        if(DistanceBetween(astroids[i].x, astroids[i].y, ship.lazer[j].x, ship.lazer[j].y) < astroids[i].ast_r + 15){
                fxHit.play();
                //remove lazer
                ship.lazer.splice(j,1);
                //remove astroids
                destroyAstroid(i);
                break;
            }
            
        }
        
    }

    //collision detection 
    if(!exploding){
        if(ship.blinkNumber ==0 ){
            for (let i = 0; i < astroids.length; i++) {
            if(DistanceBetween(ship.x, ship.y, astroids[i].x, astroids[i].y)  < (ship.radius + astroids[i].ast_r) ){
                kabooom();
                destroyAstroid(i);
                break;
            }
            

        }
        
        }
    // rotate ship              
        ship.angle += ship.rot;
    //move ship
        ship.x += ship.thrust.x;
        ship.y += ship.thrust.y;
    } else{
        ship.exploding_time--;
    }

    if(ship.exploding_time == 0){
        exploding = false;
        if(lives_left == 0){
            if(score > highScore)
                highScore = score;
            window.alert("Game over");
            newGame();
        }else ship = newShip();

    }




    // looping ship withing canvas
    if(ship.x < 0 - ship.radius){
        ship.x = canvas.width + ship.radius; 
    }else if(ship.x > canvas.width + ship.radius){
        ship.x = 0 + ship.radius;    
    }
    if(ship.y < 0 - ship.radius){
        ship.y = canvas.height + ship.radius; 
    }else if(ship.y > canvas.height + ship.radius){
        ship.y = 0 + ship.radius;    
    }

    // move astroids
    for (let i = 0; i < astroids.length; i++) {
        astroids[i].x += astroids[i].x_velocity;
        astroids[i].y += astroids[i].y_velocity;

        // loop at edges
        if(astroids[i].x < 0 - astroids[i].ast_r){
            astroids[i].x = canvas.width + astroids[i].ast_r; 
        }else if(astroids[i].x > canvas.width + astroids[i].ast_r){
            astroids[i].x = 0 + astroids[i].ast_r;    
        }
        if(astroids[i].y < 0 - astroids[i].ast_r){
            astroids[i].y = canvas.height + astroids[i].ast_r; 
        }else if(astroids[i].y > canvas.height + astroids[i].ast_r){
            astroids[i].y = 0 + astroids[i].ast_r;    
        }  
    }
        ctx.closePath();
        ctx.stroke();
}