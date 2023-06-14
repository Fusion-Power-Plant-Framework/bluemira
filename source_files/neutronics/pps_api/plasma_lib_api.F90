module pps_lib_api
  use parametric_plasma_source
  use precision_params
  use iso_c_binding
  implicit none
  
  private
  
  ! Variables that need to be initialised 
  integer, parameter :: kb = 8 
  real(dknd) :: a(1:2**kb)
  real(dknd) :: s(1:2**kb)
  
  type(c_ptr) :: seed_ptr
  
  public :: c_parametric_plasma_source, c_init_plasma
  
  contains

  ! Subroutine to sample source
  subroutine c_parametric_plasma_source(c_mode,         &
                                        c_temp,         &
                                        c_major_r,      &
                                        c_minor_r,      &
                                        c_elongation,   &
                                        c_triang,       &
                                        c_radial_shft,  &
                                        c_peaking_fctr, &
                                        c_vertical_shft,&
                                        c_start_angle,  &
                                        c_range_angle,  &
                                        c_seed,         &
                                        c_TT_channels,  &
                                        c_ratio_1,      &
                                        c_ratio_2,      &
                                        c_ratio_3,      &
                                        c_decay_5He,    &
                                        c_prt_type,     &
                                        c_prt_wgt,      &
                                        c_prt_tme,      &
                                        c_prt_erg,      &
                                        c_prt_xxx,      &
                                        c_prt_yyy,      &
                                        c_prt_zzz       &
                                        )             &
                                        bind(c, name='c_parametric_plasma_source')
    ! Input parameters
    integer(c_int), intent(in), value :: c_mode
    real(c_double), intent(in), value :: c_temp
    real(c_double), intent(in), value :: c_major_r
    real(c_double), intent(in), value :: c_minor_r
    real(c_double), intent(in), value :: c_elongation  
    real(c_double), intent(in), value :: c_triang
    real(c_double), intent(in), value :: c_radial_shft
    real(c_double), intent(in), value :: c_peaking_fctr
    real(c_double), intent(in), value :: c_vertical_shft
    real(c_double), intent(in), value :: c_start_angle 
    real(c_double), intent(in), value :: c_range_angle
    integer(c_int), intent(in), value :: c_TT_channels
    real(c_double), intent(in), value :: c_ratio_1
    real(c_double), intent(in), value :: c_ratio_2
    real(c_double), intent(in), value :: c_ratio_3
    integer(c_int), intent(in), value :: c_decay_5He  
    
    ! Particle parameters
    integer(c_int), intent(inout) :: c_prt_type   
    real(c_double), intent(inout) :: c_prt_wgt
    real(c_double), intent(inout) :: c_prt_tme
    real(c_double), intent(inout) :: c_prt_erg
    real(c_double), intent(inout) :: c_prt_xxx
    real(c_double), intent(inout) :: c_prt_yyy
    real(c_double), intent(inout) :: c_prt_zzz
    
    ! Seed for random number generator
    type(c_ptr) :: c_seed
    
    seed_ptr = c_seed

    call sample_plasma(c_mode,         &
                       c_temp,         &
                       c_major_r,      &
                       c_minor_r,      &
                       c_elongation,   &
                       c_triang,       &
                       c_radial_shft,  &
                       c_peaking_fctr, &
                       c_vertical_shft,&
                       c_start_angle,  &
                       c_range_angle,  &
                       rand_func,      &
                       c_TT_channels,  &
                       c_ratio_1,      &
                       c_ratio_2,      &
                       c_ratio_3,      &
                       c_decay_5He,    &
                       kb,             &
                       a,              &
                       s,              &
                       c_prt_type,     &
                       c_prt_wgt,      &
                       c_prt_tme,      &
                       c_prt_erg,      &
                       c_prt_xxx,      &
                       c_prt_yyy,      &
                       c_prt_zzz       &
                       )
    
  end subroutine c_parametric_plasma_source 
  
  ! Routine to initialise the a and s arrays, should be called before transport starts. 
  subroutine c_init_plasma(c_minor_r, c_peaking_fctr) bind(c, name='c_init_plasma')
    real(c_double), intent(in), value :: c_minor_r
    real(c_double), intent(in), value :: c_peaking_fctr
    
    call set_a_and_s(c_minor_r, c_peaking_fctr, kb, a, s)
  
  end subroutine c_init_plasma
  
  ! random number function which wraps the OpenMC random number generator
  function rand_func() result(rand_num)
    real(dknd) :: rand_num
    interface
      real (c_double) function prn_c(seed) bind(c)
        use iso_c_binding
        type(c_ptr) :: seed
      end function prn_c
    end interface
    
    rand_num = prn_c(seed_ptr)    
  
  end function rand_func
  
                                        
end module pps_lib_api