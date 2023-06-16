// OpenMc Modules
#include "openmc/particle_data.h"
#include "openmc/capi.h"
#include "openmc/cell.h"
#include "openmc/error.h"
#include "openmc/geometry.h"
#include "openmc/particle.h"
#include "openmc/position.h"
#include "openmc/random_lcg.h"
#include "openmc/source.h"

// cR2S Modules
#include "plasma_lib_api.h"

// C++ Modules
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <iostream>
#include <string>

// Name spaces
using namespace openmc;
using namespace std;

// Custom source class
class parametric_plasma_source : public openmc::Source{
  public:
    parametric_plasma_source(int mode,
                             double temp,
                             double major_r,
                             double minor_r,
                             double elongation,
                             double triang,
                             double radial_shft,
                             double peaking_fctr,
                             double vertical_shft,
                             double start_angle,
                             double range_angle,
                             int TT_channels,
                             double ratio_1,
                             double ratio_2,
                             double ratio_3,
                             int decay_5He
                             ) : mode_(mode),
                                 temp_(temp),
                                 major_r_(major_r),
                                 minor_r_(minor_r),
                                 elongation_(elongation),
                                 triang_(triang),
                                 radial_shft_(radial_shft),
                                 peaking_fctr_(peaking_fctr),
                                 vertical_shft_(vertical_shft),
                                 start_angle_(start_angle),
                                 range_angle_(range_angle),
                                 TT_channels_(TT_channels),
                                 ratio_1_(ratio_1),
                                 ratio_2_(ratio_2),
                                 ratio_3_(ratio_3),
                                 decay_5He_(decay_5He)
                                 {}

  static std::unique_ptr<parametric_plasma_source> from_string(std::string parameters){

      // Local variables for passing parameters
      std::unordered_map<std::string, std::string> parameter_mapping;
      std::stringstream ss(parameters);
      std::string parameter;

      // Set defaults
      std::unordered_map<std::string, double> input_doubles;
      std::unordered_map<std::string, int> input_ints;

      input_doubles["temperature"] = 0.0;
      input_doubles["major_r"] = 0.0;
      input_doubles["minor_r"] = 0.0;
      input_doubles["elongation"] = 0.0;
      input_doubles["triangulation"] = 0.0;
      input_doubles["radial_shift"] = 0.0;
      input_doubles["peaking_factor"] = 0.0;
      input_doubles["vertical_shift"] = 0.0;
      input_doubles["start_angle"] = 0.0;
      input_doubles["angle_range"] = 0.0;
      input_doubles["TT_ratio_1"] = 0.0;
      input_doubles["TT_ratio_2"] = 0.0;
      input_doubles["TT_ratio_3"] = 0.0;

      input_ints["mode"] = 0;
      input_ints["TT_channels"] = 0;
      input_ints["decay_5He"] = 0;



      while (std::getline(ss, parameter,',')){
          parameter.erase(0, parameter.find_first_not_of(' '));
          std::string key = parameter.substr(0, parameter.find_first_of('='));
          std::string value = parameter.substr(parameter.find_first_of('=') + 1, parameter.length());
          parameter_mapping[key] = value;
          if(input_doubles.count(key) == 0 && input_ints.count(key) == 0) cout << "Warning source parameter " << key << " not recognised" << endl;

      }

      // User defined parameters
      for (auto& input_param: input_doubles){
          if(parameter_mapping.count(input_param.first) !=0) input_param.second = std::stod(parameter_mapping[input_param.first]);
      }
      for (auto& input_param: input_ints){
          if(parameter_mapping.count(input_param.first) !=0) input_param.second = std::stoi(parameter_mapping[input_param.first]);
      }

      // Call the init routine
      c_init_plasma(input_doubles["minor_r"],input_doubles["peaking_factor"]);

      return std::make_unique<parametric_plasma_source>(input_ints["mode"],
                                                        input_doubles["temperature"],
                                                        input_doubles["major_r"],
                                                        input_doubles["minor_r"],
                                                        input_doubles["elongation"],
                                                        input_doubles["triangulation"],
                                                        input_doubles["radial_shift"],
                                                        input_doubles["peaking_factor"],
                                                        input_doubles["vertical_shift"],
                                                        input_doubles["start_angle"],
                                                        input_doubles["angle_range"],
                                                        input_ints["TT_channels"],
                                                        input_doubles["TT_ratio_1"],
                                                        input_doubles["TT_ratio_2"],
                                                        input_doubles["TT_ratio_3"],
                                                        input_ints["decay_5He"]
                                                        );

  }

  // Main function called from OpenMC to return a starting particle site
  openmc::SourceSite sample(uint64_t* seed) const{

      //Source site
      SourceSite site;

      // Sample random direction
      double rand = openmc::prn(seed);
      double rant = openmc::prn(seed) *2.0*M_PI;
  	  double www = (rand - 0.5) * 2.0;
      double uuu = std::sqrt(1.-std::pow(www,2))*cos(rant);
      double vvv = std::sqrt(1-std::pow(www,2))*sin(rant);
      double xxx = 0.0;
      double yyy = 0.0;
      double zzz = 0.0;
      double erg = 0.0;
      double wgt = 1.0;
      double tme = 0.0;
      int p_type = 0;

      //Call plasma source routine
      c_parametric_plasma_source(this->mode_,
                                 this->temp_,
                                 this->major_r_,
                                 this->minor_r_,
                                 this->elongation_,
                                 this->triang_,
                                 this->radial_shft_,
                                 this->peaking_fctr_,
                                 this->vertical_shft_,
                                 this->start_angle_,
                                 this->range_angle_,
                                 seed,
                                 this->TT_channels_,
                                 this->ratio_1_,
                                 this->ratio_2_,
                                 this->ratio_3_,
                                 this->decay_5He_,
                                 &p_type,
                                 &wgt,
                                 &tme,
                                 &erg,
                                 &xxx,
                                 &yyy,
                                 &zzz
                                 );

      // Set the SourceSite type
      site.particle = ParticleType::neutron;
      site.r = Position(xxx,yyy,zzz);
      site.E = erg * 1e6; // convert from MeV -> eV
      site.time = tme;
      site.wgt = wgt;

      // Set direction
      site.u = Direction(uuu,vvv,www);

      return site;


  }




  private:
    int mode_;
    double temp_;
    double major_r_;
    double minor_r_;
    double elongation_;
    double triang_;
    double radial_shft_;
    double peaking_fctr_;
    double vertical_shft_;
    double start_angle_;
    double range_angle_;
    int TT_channels_;
    double ratio_1_;
    double ratio_2_;
    double ratio_3_;
    int decay_5He_;
    uint64_t* seed_;



};

extern "C" std::unique_ptr<parametric_plasma_source> openmc_create_source(std::string parameters){

  return parametric_plasma_source::from_string(parameters);

}
