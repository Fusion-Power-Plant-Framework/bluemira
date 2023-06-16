module precision_params

  integer, parameter, public :: iknd = selected_int_kind(9)
  integer, parameter, public :: dknd = selected_real_kind(15)

end module precision_params

module parametric_plasma_source
  use precision_params
  implicit none

! Module based on the original MCNP parametric plasma source but MCNP routines have been replaced with dummy routines and variables

!  RDUM(1):
!  RDUM(2):
!  RDUM(3):
!  RDUM(4):
!  RDUM(5):
!  RDUM(6):
!  RDUM(7):
!  RDUM(8):
!  RDUM(9):
!  RDUM(10):
!  RDUM(11):
!  (         Parameters rdum(10) and rdum(11) should be used if only a section of
!   the tokamak is modelled with reflecting planes at each end. The weight of
!   the starting particles is adjusted accordingly. Set rdum(10) = 0 and
!   rdum(11) = 360.0 if a full tokamak is modelled.)
!
!  1(1): number of valid cell numbers to follow
!  IDUM(2) to IDUM(IDUM(1)+1) = valid source cells

!new version for MCNP6
! AT modified to support idum(1) as selector.
!IDUM(1) = calls this subroutine. IDUM(2) = number of source cells to follow.
!IDUM(3 - ) = source cells.
!RDUMs are standard.
! THIS LINE CHANGED
!      DO I=1,IDUM(1)
!         pbl%i%icl = NAMCHG(1,IDUM(I+1))
! to
!      DO I=1,IDUM(2)
!         pbl%i%icl = NAMCHG(1,IDUM(I+2))


      ! The T+T case has three channels (Casey et al., 2012):
      ! t+t -> (1) 4He + 2n; (2) 5He(GS) + n; (3) 5He(ES) + n
      ! These occur in different ratios, which are communicated via the  RDUM() variable
      ! channels (2) and (3) are triggered with TT_channels = 100, else only channel (1)
      ! ratio (1) = RDUM(15) ratio (2) = RDUM(16) ratio (3) = RDUM(17)
      ! RDUM(18) defines, whether the decay of the 5He nucleus is treated

contains

subroutine sample_plasma(mode,         &
                         temp,         &
                         major_r,      &
                         minor_r,      &
                         elongation,   &
                         triang,       &
                         radial_shft,  &
                         peaking_fctr, &
                         vertical_shft,&
                         start_angle,  &
                         range_angle,  &
                         rang,         &
                         TT_channels,  &
                         ratio_1,      &
                         ratio_2,      &
                         ratio_3,      &
                         decay_5He,    &
                         kb,           &
                         a_array,      &
                         s_array,      &
                         prt_type,     &
                         prt_wgt,      &
                         prt_tme,      &
                         prt_erg,      &
                         prt_xxx,      &
                         prt_yyy,      &
                         prt_zzz       &
                         )

  ! Incoming parameters
  integer, intent(in) ::  mode             ! 1.0 for D-D plasma( mean nuetron energy 2.45 MeV),2 for D-T plasma assumed( mean neutron energy 14.1) and 3 for T-T plasma
  real(dknd), intent(in) ::  temp             ! Temperature of plasma in KeV
  real(dknd), intent(in) ::  major_r          ! Plasma major radius (cm)
  real(dknd), intent(in) ::  minor_r          ! Plasma minor radius (cm)
  real(dknd), intent(in) ::  elongation       ! Elongation
  real(dknd), intent(in) ::  triang           ! Triangularity
  real(dknd), intent(in) ::  radial_shft      ! Plasma radial shift (cm)
  real(dknd), intent(in) ::  peaking_fctr     ! Plasma peaking factor
  real(dknd), intent(in) ::  vertical_shft    ! Plasma vertical shift(cm)(+ = up)
  real(dknd), intent(in) ::  start_angle      ! Start of angular extent (degrees)
  real(dknd), intent(in) ::  range_angle      ! Range of angular extent (degrees)
  integer, intent(in) ::  TT_channels      ! =100 to include channels (2) 5He(GS) + n; (3) 5He(ES) + n, else only channel (1)
  real(dknd), intent(in) :: ratio_1           ! Ratio for the first reaction, (1) 4He + 2n
  real(dknd), intent(in) :: ratio_2           ! Ratio for the second reaction, (2) 5He(GS) + n
  real(dknd), intent(in) :: ratio_3           ! Ratio for the second reaction, (3) 5He(ES) + n
  integer, intent(in) :: decay_5He            ! defines, whether the decay of the 5He nucleus is treated 0 - Not included, 1 - included
  integer, intent(in) :: kb                   ! Use to set size of a and s arrays
  real(dknd), intent(in) :: a_array(1:2**kb)
  real(dknd), intent(in) :: s_array(1:2**kb)

  ! Outgoing parameters
  integer, intent(out)    :: prt_type            ! Type of particle (1 for neutron)
  real(dknd), intent(out) :: prt_wgt             ! Particle weight
  real(dknd), intent(out) :: prt_tme             ! Particle time
  real(dknd), intent(out) :: prt_erg             ! Particle energy
  real(dknd), intent(out) :: prt_xxx             ! Particle x position
  real(dknd), intent(out) :: prt_yyy             ! Particle y position
  real(dknd), intent(out) :: prt_zzz             ! Particle z position

  ! Interface for random number generator
  interface
    function rang() result(random_num)
       use precision_params
       real(dknd) :: random_num
    end function rang
  end interface

        ! dummy subroutine.  aborts job if source subroutine is missing.
  ! if nsr==USER_DEFINED_SOURCE, subroutine source must be furnished by the user.
  ! at entrance, a random set of uuu,vvv,www has been defined.  the
  ! following variables must be defined within the subroutine:
  ! xxx,yyy,zzz,icl,jsu,erg,wgt,tme and possibly ipt,uuu,vvv,www.
  ! subroutine srcdx may also be needed.

  ! .. Use Statements ..
!  use mcnp_interfaces_mod, only : expirx
!  use mcnp_debug
!  use mcnp_global
!  use mcnp_random
!  use mcnp_params, only : DKND
!  use pblcom, only : pbl
!  use tskcom
!  use mcnp_particles, only: neutron
!  use varcom, only : dbcn, nps, ion_chg,          &
!    & ion_src_a, ion_src_z, ion_src_chg
  ! Variables ion_a, ion_z, ion_zaid are contained within tskcom which is used above,
  ! therefore no longer needed from varcom

  ! TE@CCFE modified variables to match MCNP6.2 nomenclature

  real(dknd) :: t1, t2, t3, t4, t5, qq1, qq2, fudge, mu1, mu2, mu3, s1, s2, s3, phi, R, En0, en1, En2, EHe
!  implicit real(dknd) (a-h,o-z)
  character*60 string(100)
  real TestErg
  logical :: found
!  integer :: cells(1:idum(2))

!  ! Find cell indexes of cells named on the idum card
!  do i = 1,idum(2)
!    cells(i) = namchg(1,IDUM(I+2))
!  end do

   ! modified model for 360 degree, generalised source cell
   ! problem
   !
   !       integer order
   !
   ! *** set type of particle (=1 for neutrons)
      prt_type=1
   ! *** set cell number of source as that input in source card
   !      icl=idum(1)
   ! *** set surface of departure (=0 if start point is not on a surface)
   !   pbl%i%jsu=0
   ! *** set statistical weight
      prt_wgt=range_angle/360.0
   ! *** set time of particle production
      prt_tme=0.0
   ! *** calculate energy of particle
 100  t1=2.*rang()-1.
      t2=t1**2+rang()**2
      if(t2.gt.1.0.or.t2.eq.0.0) goto 100

          !Generate another pair of random numbers for FUSION v,
          !since the velocity components should be independent.

   ! *** set particle energy and ion temperature from data in rdum array
      if(mode.eq.1)then
         qq1=sqrt(2.45)
         fudge = 1000.0
      else if(mode.eq.2)then
         qq1=sqrt(14.1)
         fudge=1237.4

! ********** addition of the TT reaction ********************************************
      else if(mode.eq.3)then
      ! The T+T case has three channels (Casey et al., 2012):
      ! t+t -> (1) 4He + 2n; (2) 5He(GS) + n; (3) 5He(ES) + n
      ! These occur in different ratios, which are communicated via the  RDUM() variable
      ! channels (2) and (3) are triggered with TT_channels = 100, else only channel (1)
      ! ratio (1) = RDUM(15) ratio (2) = RDUM(16) ratio (3) = RDUM(17)
      ! RDUM(18) defines, whether the decay of the 5He nucleus is treated

      !Two random numbers needed to sample a normal distribution of THERMAL velocity
200   t3=2.*rang()-1.
      t4=t3**2+rang()**2
      if(t4.gt.1.0.or.t4.eq.0.0) goto 200

      !Randomly select the channel, by accounting for these weights
          t5=rang() !needed to select the channel

     ! executed if no individual channel ratios are entered - TT_channels = 0
          if(TT_channels.ne.(100.0)) then

     ! only the channel (1) reaction
             prt_wgt=range_angle/360.0 * 2   ! determine weight of n per TT reaction
             mu1= 4.72                       ! i.e. close to 2 neutrons / reaction
             s1 = 4.72 !An elliptical distribution (Matsuzaki et al., 2004, Fig. 3)
             phi= acos(t3) !Because of the condition for t4, this will give the correct
             !distribution of angles phi.
             R=2.*rang()-1.
             en1=mu1+s1*sin(phi)*R  !The half-circle is centred at mu1 with a radius of s1
             qq1=sqrt(en1)  ! comunicated energy (square root)

        ! executed if individual channel ratios are entered - triggered by TT_channels=100
         else if(TT_channels.eq.(100.0)) then
              if(decay_5He.eq.0) then  ! 5He decay not treated
                   prt_wgt=range_angle/360.0*(2*ratio_1+ratio_2+ratio_3)&
                   &/(ratio_1+ratio_2+ratio_3)
                   t5=t5*(2*ratio_1+ratio_2+ratio_3)
                   if(t5.le.(2*ratio_1) )then ! (1)
                           mu1= 4.72
                           s1 = 4.72 !An elliptical distribution (Matsuzaki et al., 2004, Fig. 3)
                           phi= acos(t3)
                           R=2.*rang()-1.
                           en1=mu1+s1*sin(phi)*R
                           qq1=sqrt(en1)
                   else if(t5.gt.(2*ratio_1).and.t5.le.(2*ratio_1&
                   &+ratio_2))then ! (2)
                           mu2= 8.778 !(Bogdanova et al., 2015, Eq. (10))
                           s2= 0.273  !(Bogdanova et al., 2015, Eq. (9))
                           en1 = mu2 + s2*t3*sqrt(-(2*log(t4))/t4)  ! Gaussian distribution energy of the neutron
                           !Energy is normally distributed around mu2 with a standard deviation of s2
                           qq1=sqrt(en1)
                   else if(t5.gt.(2*ratio_1+ratio_2))then ! (3)
                           mu3= 7.738 !(Bogdanova et al., 2015, Eqs. (8) & (11))
                           s3= 1.34  !(Bogdanova et al., 2015, Eq. (11))
                           en1 = mu3 + s3*t3*sqrt(-(2*log(t4))/t4)  ! Gaussian distribution energy of the neutron
                           qq1=sqrt(en1)
                   endif

              else if(decay_5He.eq.1) then  ! 5He decay also treated
                   prt_wgt=range_angle/360.0 * 2
                   t5=t5*2*(ratio_1+ratio_2+ratio_3)
                   if(t5.le.(2*ratio_1) )then ! (1)
                           mu1= 4.72
                           s1 = 4.72 !An elliptical distribution (Matsuzaki et al., 2004, Fig. 3)
                           phi= acos(t3)
                           R=2.*rang()-1.
                           en1=mu1+s1*sin(phi)*R !The half-circle is centred at mu1 with a radius of s1
                           qq1=sqrt(en1)
                   else if(t5.gt.(2*ratio_1).and.t5.le.(2*ratio_1&
                   &+ratio_2))then ! (2)
                           mu2= 8.778 !(Bogdanova et al., 2015, Eq. (10))
                           s2= 0.273  !(Bogdanova et al., 2015, Eq. (9))
                           en1 = mu2 + s2*t3*sqrt(-(2*log(t4))/t4)  ! Gaussian distribution energy of the neutron
                           qq1=sqrt(en1)
                  else if(t5.gt.(2*ratio_1+ratio_2).and.t5.le.&
                  &(2*ratio_1+ratio_2+ratio_3)) then  ! (3)
                           mu3= 7.738 !(Bogdanova et al., 2015, Eqs. (8) & (11))
                           s3= 1.34  !(Bogdanova et al., 2015, Eq. (11))
                           en1 = mu3 + s3*t3*sqrt(-(2*log(t4))/t4)  ! Gaussian distribution energy of the neutron
                           qq1=sqrt(en1)
                  else if(t5.gt.(2*ratio_1+ratio_2+ratio_3).and&
                  &.t5.le.(2*ratio_1+2*ratio_2+ratio_3)) then  ! (2) - secondary)
                           mu2= 8.778 !(Bogdanova et al., 2015, Eq. (10))
                           s2= 0.273  !(Bogdanova et al., 2015, Eq. (9))
                           En2 = mu2 + s2*t3*sqrt(-(2*log(t4))/t4)  ! Gaussian distribution energy of the 1st neutron
                           EHe = 10.534 - (mu2 + s2*t3*&
                           &sqrt(-(2*log(t4))/t4)) ! Sqrt of Gaussian distribution energy of 5He
                           En0 = 0.6384                ! neutron energy due to 5He decay in CMS
                           en1 = EHe/4 + En0 + sqrt(EHe/4*En0)*& !(Bogdanova et al., 2015, Eq. (14))
                           &(2*rang()-1)                 ! Vector summation of 5He and neutron CMS velocities
                           qq1=sqrt(en1)
                  else if(t5.gt.(2*ratio_1+2*ratio_2+ratio_3))then ! (3) - secondary
                           mu3= 7.738 !(Bogdanova et al., 2015, Eqs. (8) & (11))
                           s3= 1.34  !(Bogdanova et al., 2015, Eq. (11))
                           EHe = abs(9.264 - (mu3 + s3*t3*&
                           &sqrt(-(2*log(t4))/t4)))  ! Sqrt of Gaussian distribution energy of 5He
                           En0 = 1.6544                ! neutron energy due to 5He decay in CMS
                           en1 = abs(EHe/4 + En0 + &   !(Bogdanova et al., 2015, Eq. (14))
                           &sqrt(EHe/4*En0)*(2*rang()-1))
                           qq1=sqrt(en1)
                  endif

              endif

           endif
           fudge = 1000.0
	   ! Absolute upper bound for neutron energy 9.443 MeV, smooth ending
	   !Moved inside of loop
       if((qq1**2-9.0).gt.(rang()*0.443)) goto 100
       qq2=sqrt(temp/fudge)/2.0
	   TestErg = (qq1+qq2*t1*sqrt(-log(t2)/t2))**2
	   if (TestErg < 0 )  goto 100
	   if (isnan(TestErg)) goto 100
      endif !mode DD/DT/TT if test


      qq2=sqrt(temp/fudge)/2.0
   !      qq2=sqrt(rdum(2)/1000.0)/2.0
   ! *** define energy of particle
      prt_erg=(qq1+qq2*t1*sqrt(-log(t2)/t2))**2
      !write(723,*) pbl%r%erg, qq1, qq2, t1,t1
   ! *** call D-shape distribution of the particles in the plasma
!
!cdd      call srgnt(xxx,yyy,zzz,idum,rdum)
      call srgnt(major_r,minor_r,elongation,triang, radial_shft, peaking_fctr, vertical_shft, start_angle, range_angle, kb, a_array, s_array, prt_xxx, prt_yyy, prt_zzz, rang)
   !    find the number of the cell containing the starting point
   !    IDUM(*) is the list of possible source cell numbers
      ! J = 0
      ! pbl%i%icl = 0
      ! call findlv
      ! found = .false.
      ! do i = 1, IDUM(2)
      !   if(pbl%i%icl == cells(i)) then
      !     found = .true.
      !     exit
      !   end if
      ! end do
      ! DO I=1,IDUM(2)
      !   pbl%i%icl = NAMCHG(1,IDUM(I+2))
      !   CALL CHKCEL(pbl%i%icl,2,J)
      !   IF (J.EQ.0) GOTO 999
      ! ENDDO
 999  CONTINUE
      !if(pbl%i%icl.eq.0.or..not.found)goto 100
  return
end subroutine sample_plasma
   ! ---------------------------------------------------------------------
!cdd      subroutine srgnt(xxx,yyy,zzz,idum,rdum)
      subroutine srgnt(rm,ap,e0,cp0,esh,epk,deltaz,start_angle,range_angle, kb, a, s, prt_xxx, prt_yyy, prt_zzz, rang)
!  use mcnp_interfaces_mod, only : expirx
!  use mcnp_debug
!  use mcnp_global
!  use mcnp_random
!  use mcnp_params, only : DKND
!  use pblcom, only : pbl
!  use tskcom
!  use mcnp_particles, only: neutron
!      implicit real(dknd) (a-h,o-z)
!
!cdd      dimension a(256),s(256),idum(50),rdum(50)
      ! Variables passed in
      real(dknd), intent(in) :: rm
      real(dknd), intent(in) :: ap
      real(dknd), intent(in) :: e0
      real(dknd), intent(in) :: cp0
      real(dknd), intent(in) :: esh
      real(dknd), intent(in) :: epk
      real(dknd), intent(in) :: deltaz
      real(dknd), intent(in) :: start_angle
      real(dknd), intent(in) :: range_angle
      integer, intent(in) :: kb
      real(dknd), intent(in) :: a(1:2**kb)
      real(dknd), intent(in) :: s(1:2**kb)

      ! Variables passed out
      real(dknd), intent(out) :: prt_xxx
      real(dknd), intent(out) :: prt_yyy
      real(dknd), intent(out) :: prt_zzz


      real(dknd) :: r, e, e1, cp, n1, fn, fn1, x, z, sint, t, qxd, qxs, estar, FI
      integer :: ip
      integer :: icall
      integer :: i, n

      ! Interface for random number generator
      interface
        function rang() result(random_num)
           use precision_params
           real(dknd) :: random_num
        end function rang
      end interface

     ! *** set geometric variables according to input of source card
     e1=0.0

   ! *** define r value of particle
 200  r=sqrt((rm-ap)*(rm-ap)+4.*rm*ap*rang())

   ! *** define e
      e=e0+e1

   ! *** use this line for a source above and below midplane
      prt_zzz=-e*ap+2.*e*ap*rang()

      z=abs(prt_zzz)
      prt_zzz=prt_zzz+deltaz
      x=r-rm
      sint=z/(ap*e)
      t=asin(sint)
      qxd=ap*cos(t+cp0*sint)
      if(x.ge.qxd) goto 200
      qxs=ap*cos(3.141592654-t+cp0*sint)
      if(x.le.qxs) goto 200
      i=2**(kb-1)
      ip=0
      do 20 n=1,kb
         i=i+ip*2**(kb-n)
         ip=+1
         estar=e
         if(z.ge.a(i)*estar) goto 20
         sint=z/(a(i)*estar)
         t=asin(sint)
         cp=cp0
         qxd=a(i)*cos(t+cp*sint)+esh*(1.-(a(i)/ap)**2)
         if(x.ge.qxd) goto 20
         qxs=a(i)*cos(3.141592654-t+cp*sint)+esh*(1.-(a(i)/ap)**2)
         if(x.le.qxs) goto 20
         ip=-1
 20   continue
      ip=max0(0,ip)
      if(rang().gt.s(i+ip)) go to 200
   !
   ! *** define the angle about z axis at which the particle is:
   ! *** 90 degrees about y axis (i.e y is always positive and x is either +ve
   ! *** of -ve)
      FI = (start_angle+range_angle* RANG())*0.017453292
      prt_xxx = R * COS(FI)
      prt_yyy = R * SIN(FI)

  return
end subroutine srgnt

subroutine set_a_and_s(ap,epk,kb, a,s)

  implicit none

  integer, intent(in) :: kb
  real(dknd), intent(in) :: ap
  real(dknd), intent(in) :: epk
  real(dknd), intent(out) :: s(1:2**kb)
  real(dknd), intent(out) :: a(1:2**kb)

  integer :: n
  integer :: n1
  real(dknd) :: fn
  real(dknd) :: fn1

    ! *** set geometric variables according to input of source card
    ! *** define some variables
    n1=2**kb
    fn1=n1
    do n=1,n1
       fn=n
       s(n)=(2.*fn1-2.*fn+1.)/(2.*fn1-1.)
       a(n)=ap*sqrt(1.-((fn1-fn+1.)/fn1)**(1./epk))
    end do

end subroutine set_a_and_s

end module parametric_plasma_source
