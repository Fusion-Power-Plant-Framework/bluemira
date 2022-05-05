
    !!! Inputs from BLUEPRINT !!!
    ! Utility inputs - do not touch
    ! DEMO1
    num%tol=1.e-10  !tolerance to be reached, in % variation at each time step
    num%dtmin=0.05d0 !min time step
    num%dtmax=0.05d0 !max time step
    num%dt=0.01d0 !time step
    num%dtinc=2.d0 !decrease of dt
    num%Ainc=1.1d0 !increase of dt
    num%test=10000. !max iteration number
    num%tolmin=10.1d0 ! multiplier of etolm that should not be overcome
    num%eopt=0.5d0 !exponent of jipperdo
    num%dtmaxmin=0.1d0 !exponent of jipperdo2
    num%capA=0.01d0 !first radial grid point for reduced grid
    num%maxA=0.d0 !diagz 0 or 1
    num%dgy=1.e-5 !Newton differential
    num%i_modeltype=1 !1 - simple gyrobohm scaling with imposed H factor, > 1, other models with H in output, 9999 TGLF
    num%i_equiltype=2 !1 - EMEQ, solve equilibrium with given q95, with sawteeth. 2- EMEQ, solve with given Ip, with sawteeth.
    num%nx=41        !number of interpolated grid points
    num%nxt=11 !number of reduced grid points
    num%nchannels=3  !leave this at 3
    num%ipedestal=1 !1 - fixed temperature pedestal. 2 - Sareelma scaling
    num%i_impmodel=1 !impurity model: 0 - fixed concentration, 1 - concentration fixed at pedestal top, then fixed density.
    num%dtmaxmax=1. !stabilizing diff for TGLF in mB2/s
    num%dtminmax=5. !tolerance above which TGLF should be always called
    num%ntglf=11 !number of tglf points, below positions
    num%xtglf(1)=0.1 !tolerance above which TGLF should be always called
    num%xtglf(2)=0.15 !tolerance above which TGLF should be always called
    num%xtglf(3)=0.2 !tolerance above which TGLF should be always called
    num%xtglf(4)=0.25 !tolerance above which TGLF should be always called
    num%xtglf(5)=0.3 !tolerance above which TGLF should be always called
    num%xtglf(6)=0.4 !tolerance above which TGLF should be always called
    num%xtglf(7)=0.5 !tolerance above which TGLF should be always called
    num%xtglf(8)=0.6 !tolerance above which TGLF should be always called
    num%xtglf(9)=0.7 !tolerance above which TGLF should be always called
    num%xtglf(10)=0.75 !tolerance above which TGLF should be always called
    num%xtglf(11)=0.8 !tolerance above which TGLF should be always called
    num%isawt=1 ! 0 for natural q profile, 1 for clamped at 1

    !geometry
    geom%A = 3.1d0  !aspect ratio
    geom%R = 8.8823d0  ! major radius in m
    geom%bt = 5.8217 !magnetic field
    geom%k = 1.7920000000000003d0 !edge elongation
    geom%d = 0.37296000000000007d0 !edge triangularity
    geom%k95 = 1.6d0 !edge elongation
    geom%d95 = 0.333d0 !edge triangularity
    geom%Ip =  17.833 !plasma current in MA : USED if equiltype=2, q95 is used if equiltype=1
    geom%q95 = 3.5 !safety factor.

    inp0%f_gw = 0.85 !pedestal top greenwald fraction
    inp0%f_gws= 0.5 !separatrix greenwald fraction

    ped%teped=  6.d0  !pedestal top temperature

    geom%counter=0.


    comp%globtau(1) = 7. !tauparticle/tauE for D, T, He, Xe, Ar
    comp%globtau(2) = 7. !tauparticle/tauE for D, T, He, Xe, Ar
    comp%globtau(3) = 7. !tauparticle/tauE for D, T, He, Xe, Ar
    comp%globtau(4) = 7. !tauparticle/tauE for D, T, He, Xe, Ar
    comp%globtau(5) = 1. !tauparticle/tauE for D, T, He, Xe, Ar

    comp%comparray=0.d0
    comp%protium = 0.d0

    comp%car = 0. !argon concentration, used if qdivt=0.
    comp%cxe = 0. !xenon concentration, if negative uses Psepplh as criterion
    comp%che = 0. !helium concentration, used if globtau(3)=0.

    comp%fuelmix = 0.5d0 !fuel mix

    comp%pradpos = 0.75d0 ! position after which radiation is counted 0. for tau and other global quantities, i.e. position after which radiation is "edge"
    comp%pradfrac = 0.6d0 ! position after which radiation is counted 0. for tau and other global quantities, i.e. position after which radiation is "edge"

    ped%tesep=0.1  !separatrix temperature
    ped%rho_t=0.94 !pedestal top position T in r/a
    ped%rho_n=0.94 !pedestal top position n in r/a
    ped%pedscal=1.1

    inp0%nbcdeff=0.3 !CD = this * PCD   units: m*MA/MW (MA/m^2 * m^3/MW)
    inp0%eccdeff=0.3 !CD = this * PCD * TE/NE !not used for now
    inp0%pheatmax=100. !max allowed power for heating+CD+fusion control
    inp0%pech=0.d0 !ech power !not used for now
    inp0%pnbi=0.d0 !nbi power
    inp0%qheat=0.d0 !nbi power
    inp0%qcd=0.d0 !nbi power
    inp0%qfus=0.d0 !nbi power
    inp0%spellet=0.d0 !pellet mass in particles of D in 10^19
    inp0%fpellet=0.5d0 !pellet frequency in Hz
    inp0%q_control=50.d0 !minimal power required for control
    inp0%fcdp=-1.d0 !ratio of PCD-Pothers over Pmax - Pothers
    inp0%fpion=0.5d0 !ratio of NBI power to ioins

    inp0%maxpauxor=20. ! maximum Paux/R allowed

    !constraints
    inp0%V_loop=-1.e6 !target loop voltage. If lower than -1.e5, dont use this
    inp0%Hfac_inp=1 !input H factor, if imodeltype > 1 this is ignored
    inp0%f_ni=0.1 !required fraction of non inductive current, if 0, dont use CD
    inp0%pfus=0. !if 0., not used (otherwise it would be controlled with Pauxheat)
    inp0%PLH=6
    comp%psepplh_inf = 1. !Psep/PLH if below this, use nbi
    comp%psepplh_sup = 1000.2d0 !Psep/PLH if above this, use Xe
    comp%psepb_q95AR = 9.2d0 !Psep B/qaR max value
    comp%psep_r = 10009.d0 !Psep/R max value
    comp%fcoreraditv = -1.d0 !ratio of Pxe / (Psep0 -Psep_crit) ; Psep0 is Palpha + Paux - Par - Pbrehm - Psync

    num%isiccir=0 !SOL routine. 0 - fit, 1- Msicci model
    comp%qdivt = 0. !divertor heat flux in MW/m^2, if 0, dont use SOL model
    comp%c_car = 10. !compression factor between sol and div: e.g. 10 means there is 10 more Argon concentration in the divertor than in the core

    inp0%gamcdothers=1.d0 !multiplies CD efficiency of everything that is not q_CD.


    !sawteeth
    inp0%chisaw=0. !sawtooth diffusibity in mB2/s, leave 0 if not to be used.
    inp0%chisawpos=-1.d0 !position where chisaw is applied. if negative, uses q=1 position

    !derivatives
    inp0%qnbi_psepfac=10. !dqnbi/d(1-Psep/PLH)
    inp0%cxe_psepfac=1.e-4 !dcxe/d(1-Psep/PLH)
    inp0%car_qdivt=1.e-5 !dcar/d(qdivt)
    inp0%contrpovs=0.d0 !Pcontrol/S_lateral
    inp0%contrpovr=0.d0 !Pcontrol/R
    !deposition locations
    inp0%x_heat(1)=0. !nbi
    inp0%x_heat(2)=0. !ech
    inp0%dx_heat(1)=0.2 !nbi
    inp0%dx_heat(2)=0.03 !ech
    inp0%x_cd(1)=0. !nbi
    inp0%x_cd(2)=0. !ech
    inp0%dx_cd(1)=0.2 !nbi
    inp0%dx_cd(2)=0.03 !ech
    inp0%x_fus(1)=0. !nbi
    inp0%x_fus(2)=0. !ech
    inp0%dx_fus(1)=0.2 !nbi
    inp0%dx_fus(2)=0.03 !ech
    inp0%x_control(1)=0. !nbi
    inp0%x_control(2)=0. !ech
    inp0%dx_control(1)=0.2 !nbi
    inp0%dx_control(2)=0.03 !ech
    inp0%nbi_energy=1000. !in keV
