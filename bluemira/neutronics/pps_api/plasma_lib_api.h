#ifdef __cplusplus
extern "C" {
#endif

  void c_parametric_plasma_source(int c_mode,
                                  double c_temp,
                                  double c_major_r,
                                  double c_minor_r,
                                  double c_elongation,
                                  double c_triang,
                                  double c_radial_shft,
                                  double c_peaking_fctr,
                                  double c_vertical_shft,
                                  double c_start_angle,
                                  double c_range_angle,
                                  uint64_t* c_rang,
                                  int c_TT_channels,
                                  double c_ratio_1,
                                  double c_ratio_2,
                                  double c_ratio_3,
                                  int c_decay_5He,
                                  int * c_prt_type,
                                  double * c_prt_wgt,
                                  double * c_prt_tme,
                                  double * c_prt_erg,
                                  double * c_prt_xxx,
                                  double * c_prt_yyy,
                                  double * c_prt_zzz
                                 );


  void c_init_plasma(double c_minor_r, double c_peaking_fctr);

#ifdef __cplusplus
}
#endif
