//#![feature(portable_simd)]
//use std::env;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::time::SystemTime;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use chrono::{DateTime, Utc};
use rayon::prelude::*;

fn update(max_rods: u32, max_pearls: u32, max_attempts: u64, when_found: &str, attempts: u64,
    total_exec_time: f64, speed: f64) {


    let writestring = format!("{},{},{},{},{},{},{}", max_rods, max_pearls, max_attempts, when_found, attempts, total_exec_time, speed);

    let mut file = File::create("dream.txt")
        .expect("Could not open the dream.txt file to write.");

    // Open a file in write-only mode, returns `io::Result<File>`
    file.write(writestring.as_bytes())
        .expect("Could not write to the dream.txt file.");

    println!("Attempts:{}, Rods: {}, Pearls: {}, Speed: {} attempts/sec", attempts, max_rods, max_pearls, speed);

}

fn main() {

    const PEARLTHRESHOLD: f64 = 20.0;
    const PEARLRANGE: f64 = 423.0;
    const GOLDTRADES: u32 = 262;
    const DREAMPEARLS: u32 = 42;
    const DREAMRODS: u32 = 211;

    let mut max_rods;
    let mut max_pearls;
    let mut max_attempts;

    let mut attempts;
    let mut when_found;
    let mut total_exec_time;
    let mut speed;

    let mut batch_size: u64 = 100_000;


    // load previous data from file
    let contents = fs::read_to_string("dream.txt")
        .expect("Something went wrong reading the file");

    let split_string: Vec<&str> = contents.split(",").collect();

    max_rods = split_string[0].parse::<u32>().unwrap();
    max_pearls = split_string[1].parse::<u32>().unwrap();
    max_attempts = split_string[2].parse::<u64>().unwrap();
    when_found = String::from(split_string[3]);
    attempts = split_string[4].parse::<u64>().unwrap();
    total_exec_time = split_string[5].parse::<f64>().unwrap();

    // main loop
    loop {
        //let mut frng = fastrand::seed(rng.gen());

        // start timer
        let now = SystemTime::now();

        if attempts >= 500_000_000 {
            batch_size = 500_000_000;
        } else if attempts >= 100_000_000 {
            batch_size = 100_000_000;
        } else if attempts >= 1_000_000 {
            batch_size = 1_000_000;
        }

        
        let rods: Vec<u32> = (0..batch_size).into_par_iter()
                                .map_init(
                                    || Xoshiro256PlusPlus::from_entropy(),
                                    |rng, _x| (rng.gen::<u128>(), rng.gen::<u128>(), rng.gen::<u64>() & 0x1FFFF_FFFF_FFFFu64),
                                )
                                .map(|x| x.0.count_ones() + x.1.count_ones() + x.2.count_ones())
                                .filter(|x| *x >= max_rods || *x >= DREAMRODS)
                                .collect();

        let mut pearls: Vec<(u32, u32)> = rods.clone().into_par_iter()
                                        .map_init(
                                            || Xoshiro256PlusPlus::from_entropy(),
                                            |rng, x| (x, (0..GOLDTRADES).into_iter()
                                                            .map(|_y| rng.gen_bool(PEARLTHRESHOLD / PEARLRANGE) as u32)
                                                            .sum()),
                                        )
                                        .filter(|x| (x.1 >= max_pearls || x.1 >= DREAMPEARLS))
                                        .collect();


        let new_len = pearls.len();
        
        if new_len > 0 {
            let new_record;

            if new_len == 1 {
                new_record = pearls.get(0).expect("Expected 1 item in list of length 1.");
            } else {
                println!("{} in batch higher than record.", new_len);
                pearls.sort_by(|b, a| a.0.partial_cmp(&b.0).unwrap());
                pearls.sort_by(|b, a| a.1.partial_cmp(&b.1).unwrap());

                new_record = pearls.get(0).expect("Expected 1 item in list of length greater than 1.");
            }

            

            let utc: DateTime<Utc> = Utc::now();
            when_found = utc.format("%F %T").to_string();
            max_attempts = attempts;
            max_rods = new_record.0;
            max_pearls = new_record.1;

            println!("New record: {} rods, {} pearls", max_rods, max_pearls);
        }        
        
        attempts += batch_size;                               
                                        
        //dbg!(&rods);
        //dbg!(pearls);

        //std::process::exit(0);


        // stop timer
        match now.elapsed() {
            Ok(elapsed) => {
                let seconds = elapsed.as_secs_f64();
                speed = batch_size as f64 / seconds;
                total_exec_time += seconds;
                update(max_rods, max_pearls, max_attempts, &when_found, attempts, total_exec_time, speed);
            }
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }
}
