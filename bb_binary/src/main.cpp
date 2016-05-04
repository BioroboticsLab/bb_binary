# include "csv.h"

int main(){
  io::CSVReader<3> in("ram.csv");
  std::string vendor; int size; double speed;
  while(in.read_row(vendor, size, speed)){
    // do stuff with the data
  }
}
